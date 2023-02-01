import math
import os
import time
import random
import pickle
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import torch

from config.config import cfg
from models.loss_utils import TargetDiscrimLoss, SourceDiscrimLoss, ConcatenatedEMLoss
from utils.neptune import log_list
from utils.utils import to_cuda, accuracy_for_each_class, accuracy, AverageMeter
from .base_solver import BaseSolver


class SymmNetsV1Solver(BaseSolver):
    def __init__(self, net, dataloaders, logger, **kwargs):
        super(SymmNetsV1Solver, self).__init__(net, dataloaders, logger, **kwargs)
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.TargetDiscrimLoss = TargetDiscrimLoss(num_classes=self.num_classes).cuda()
        self.SourceDiscrimLoss = SourceDiscrimLoss(num_classes=self.num_classes).cuda()
        self.ConcatenatedEMLoss = ConcatenatedEMLoss(num_classes=self.num_classes).cuda()
        self.feature_extractor = self.net['feature_extractor']
        self.classifier = self.net['classifier']

        if cfg.RESUME != '':
            resume_dict = torch.load(cfg.RESUME)
            self.net['feature_extractor'].load_state_dict(resume_dict['feature_extractor_state_dict'])
            self.net['classifier'].load_state_dict(resume_dict['classifier_state_dict'])
            self.best_prec1 = resume_dict['best_prec1']
            self.epoch = resume_dict['epoch']

    def solve(self):
        stop = False
        while not stop:
            stop = self.complete_training()
            self.update_network()
            acc = self.test()
            if acc > self.best_prec1:
                self.best_prec1 = acc
                log = open(os.path.join(self.opt.SAVE_DIR, 'log.txt'), 'a')
                log.write("      Best Acc so far: %3f" % (self.best_prec1))
                log.close()
                if self.opt.TRAIN.SAVING:
                    self.save_ckpt()
            self.epoch += 1
        self.save_ckpt('last')
        self.create_tsne_plot()

    def update_network(self, **kwargs):
        stop = False
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
        self.iters_per_epoch = len(self.train_data['target']['loader'])
        iters_counter_within_epoch = 0
        data_time = AverageMeter()
        batch_time = AverageMeter()
        classifier_loss = AverageMeter()
        feature_extractor_loss = AverageMeter()
        prec1_fs = AverageMeter()
        prec1_ft = AverageMeter()
        self.feature_extractor.train()
        self.classifier.train()
        end = time.time()
        if self.opt.TRAIN.PROCESS_COUNTER == 'epoch':
            lam = 2 / (1 + math.exp(-1 * 10 * self.epoch / self.opt.TRAIN.MAX_EPOCH)) - 1
            new_lr = self.update_lr()
            if self.logger:
                log_list(self.logger, ['train\lam', 'train\lr'], [lam, new_lr])
            print('value of lam is: %3f' % (lam))
        while not stop:
            if self.opt.TRAIN.PROCESS_COUNTER == 'iteration':
                lam = 2 / (1 + math.exp(-1 * 10 * self.iters / (self.opt.TRAIN.MAX_EPOCH * self.iters_per_epoch))) - 1
                print('value of lam is: %3f' % (lam))
                new_lr = self.update_lr()
                if self.logger:
                    log_list(self.logger, ['train\lam', 'train\lr'], [lam, new_lr])
            source_data, source_gt = self.get_samples('source')
            target_data, _ = self.get_samples('target')
            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            target_data = to_cuda(target_data)
            data_time.update(time.time() - end)

            # Get features and classification
            feature_source = self.feature_extractor(source_data)
            output_source = self.classifier(feature_source)
            feature_target = self.feature_extractor(target_data)
            output_target = self.classifier(feature_target)

            # Calculate losses
            loss_task_fs = self.CELoss(output_source[:, :self.num_classes], source_gt)
            loss_task_ft = self.CELoss(output_source[:, self.num_classes:], source_gt)
            if self.opt.MCDALNET.DISTANCE_TYPE != 'SourceOnly':
                loss_discrim_source = self.SourceDiscrimLoss(output_source)
                loss_discrim_target = self.TargetDiscrimLoss(output_target)
                loss_summary_classifier = loss_task_fs + loss_task_ft + loss_discrim_source + loss_discrim_target
            else:
                loss_summary_classifier = loss_task_fs + loss_task_ft

            self.optimizer_classifier.zero_grad()
            loss_summary_classifier.backward(retain_graph=True)
            self.optimizer_classifier.step()
            classifier_loss.update(loss_summary_classifier, source_data.size()[0])
            del loss_summary_classifier

            output_source = self.classifier(feature_source)
            output_target = self.classifier(feature_target)

            source_gt_for_ft_in_fst = source_gt + self.num_classes
            if self.opt.MCDALNET.DISTANCE_TYPE != 'SourceOnly':
                loss_confusion_source = 0.5 * self.CELoss(output_source, source_gt) + 0.5 * self.CELoss(output_source,
                                                                                                        source_gt_for_ft_in_fst)
                loss_confusion_target = 0.5 * self.SourceDiscrimLoss(output_target) + 0.5 * self.TargetDiscrimLoss(
                    output_target)
                loss_em = self.ConcatenatedEMLoss(output_target)
                loss_summary_feature_extractor = loss_confusion_source + lam * (loss_confusion_target + loss_em)
            else:
                loss_summary_feature_extractor = self.CELoss(output_source[:, :self.num_classes], source_gt)

            self.optimizer_feature_extractor.zero_grad()
            loss_summary_feature_extractor.backward()
            self.optimizer_feature_extractor.step()

            feature_extractor_loss.update(loss_summary_feature_extractor, source_data.size()[0])
            del loss_summary_feature_extractor

            prec1_fs.update(accuracy(output_source[:, :self.num_classes], source_gt), source_data.size()[0])
            prec1_ft.update(accuracy(output_source[:, self.num_classes:], source_gt), source_data.size()[0])

            del output_target, output_source, feature_target, feature_source

            print("  Train:epoch: %d:[%d/%d], LossCla: %3f, LossFeat: %3f, AccFs: %3f, AccFt: %3f" % \
                  (self.epoch, iters_counter_within_epoch, self.iters_per_epoch, classifier_loss.avg,
                   feature_extractor_loss.avg, prec1_fs.avg, prec1_ft.avg))
            if self.logger:
                log_list(self.logger,
                         ["train\epoch", "train\iter", "train\LossCla", "train\LossFeat", "train\AccFs", "train\AccFt"],
                         [self.epoch, iters_counter_within_epoch, classifier_loss.avg,
                          feature_extractor_loss.avg, prec1_fs.avg, prec1_ft.avg])

            batch_time.update(time.time() - end)
            end = time.time()
            self.iters += 1
            iters_counter_within_epoch += 1
            if iters_counter_within_epoch >= self.iters_per_epoch:
                log = open(os.path.join(self.opt.SAVE_DIR, 'log.txt'), 'a')
                log.write("\n")
                log.write("  Train:epoch: %d:[%d/%d], LossCla: %3f, LossFeat: %3f, AccFs: %3f, AccFt: %3f" % \
                          (self.epoch, iters_counter_within_epoch, self.iters_per_epoch, classifier_loss.avg,
                           feature_extractor_loss.avg, prec1_fs.avg, prec1_ft.avg))
                log.close()
                if self.logger:
                    log_list(self.logger,
                             ["train\epoch", "train\iter", "train\LossCla", "train\LossFeat", "train\AccFs",
                              "train\AccFt"],
                             [self.epoch, iters_counter_within_epoch, classifier_loss.avg,
                              feature_extractor_loss.avg, prec1_fs.avg, prec1_ft.avg])
                stop = True

    def test(self):
        self.feature_extractor.eval()
        self.classifier.eval()
        prec1_fs = AverageMeter()
        prec1_ft = AverageMeter()
        counter_all_fs = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_all_ft = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_acc_fs = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        counter_acc_ft = torch.FloatTensor(self.opt.DATASET.NUM_CLASSES).fill_(0)
        y_true, y_pred = [], []

        for i, (input, target) in enumerate(self.test_data['loader']):
            input, target = to_cuda(input), to_cuda(target)
            with torch.no_grad():
                feature_test = self.feature_extractor(input)
                output_test = self.classifier(feature_test)
                _, pred = output_test[:, :self.num_classes].topk(1, 1, True, True)
                y_pred += pred.cpu().detach().tolist()
                y_true += target.cpu().detach().tolist()

            if self.opt.EVAL_METRIC == 'accu':
                prec1_fs_iter = accuracy(output_test[:, :self.num_classes], target)
                prec1_ft_iter = accuracy(output_test[:, self.num_classes:], target)
                prec1_fs.update(prec1_fs_iter, input.size(0))
                prec1_ft.update(prec1_ft_iter, input.size(0))
                if i % self.opt.PRINT_STEP == 0:
                    print("  Test:epoch: %d:[%d/%d], AccFs: %3f, AccFt: %3f" % \
                          (self.epoch, i, len(self.test_data['loader']), prec1_fs.avg, prec1_ft.avg))
            elif self.opt.EVAL_METRIC == 'accu_mean':
                prec1_ft_iter = accuracy(output_test[:, self.num_classes:], target)
                prec1_ft.update(prec1_ft_iter, input.size(0))
                counter_all_fs, counter_acc_fs = accuracy_for_each_class(output_test[:, :self.num_classes], target,
                                                                         counter_all_fs, counter_acc_fs)
                counter_all_ft, counter_acc_ft = accuracy_for_each_class(output_test[:, self.num_classes:], target,
                                                                         counter_all_ft, counter_acc_ft)
                if i % self.opt.PRINT_STEP == 0:
                    print("  Test:epoch: %d:[%d/%d], Task: %3f" % \
                          (self.epoch, i, len(self.test_data['loader']), prec1_ft.avg))
            else:
                raise NotImplementedError
        acc_for_each_class_fs = counter_acc_fs / counter_all_fs
        acc_for_each_class_ft = counter_acc_ft / counter_all_ft
        log = open(os.path.join(self.opt.SAVE_DIR, 'log.txt'), 'a')
        log.write("\n")
        
        # Classification report
        cr = classification_report(y_true, y_pred, labels=range(self.num_classes), digits=6)
        if self.logger:
            self.logger[f"classification_report"] = cr
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        if self.logger:
            self.logger[f"plots/cm"].upload(plt.gcf())

        if self.opt.EVAL_METRIC == 'accu':
            log.write(
                "                                                          Test:epoch: %d, AccFs: %3f, AccFt: %3f" % \
                (self.epoch, prec1_fs.avg, prec1_ft.avg))
            log.close()
            if self.logger:
                log_list(self.logger, ["test\epoch", "test\AccFs", "test\AccFt"],
                         [self.epoch, prec1_fs.avg, prec1_ft.avg])
            return max(prec1_fs.avg, prec1_ft.avg)
        elif self.opt.EVAL_METRIC == 'accu_mean':
            log.write(
                "                                            Test:epoch: %d, AccFs: %3f, AccFt: %3f" % \
                (self.epoch, acc_for_each_class_fs.mean(), acc_for_each_class_ft.mean()))
            log.write("\nClass-wise Acc of Ft:")  ## based on the task classifier.
            if self.logger:
                log_list(self.logger, ["test\epoch", "test\ClassAccFs", "test\ClassAccFt", "test\AccFt"],
                         [self.epoch, acc_for_each_class_fs.mean(), acc_for_each_class_ft.mean(), prec1_ft.avg])
            for i in range(self.opt.DATASET.NUM_CLASSES):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class_ft[i]))
                if self.logger:
                    log_list(self.logger, ["test\epoch", f"test\Class_{i:02d}_AccFs", f"test\Class_{i:02d}_AccFt"],
                             [self.epoch, acc_for_each_class_fs[i], acc_for_each_class_ft[i]])

            log.close()
            return max(acc_for_each_class_ft.mean(), acc_for_each_class_fs.mean())

    def build_optimizer(self):
        if self.opt.TRAIN.OPTIMIZER == 'SGD':  ## some params may not contribute the loss_all, thus they are not updated in the training process.
            params = [
                {'params': self.net['feature_extractor'].conv1.parameters(), 'name': 'pre-trained'},
                {'params': self.net['feature_extractor'].bn1.parameters(), 'name': 'pre-trained'},
                {'params': self.net['feature_extractor'].layer1.parameters(), 'name': 'pre-trained'},
                {'params': self.net['feature_extractor'].layer2.parameters(), 'name': 'pre-trained'},
                {'params': self.net['feature_extractor'].layer3.parameters(), 'name': 'pre-trained'},
                {'params': self.net['feature_extractor'].layer4.parameters(), 'name': 'pre-trained'},
            ]
            if self.net['feature_extractor'].bp is not None:
                params += [
                    {'params': self.net['feature_extractor'].bp.parameters(), 'name': 'new-added'},
                    #                    {'params': self.net['feature_extractor'].module.cbam.parameters(), 'name': 'new-added'},
                    #                    {'params': self.net['feature_extractor'].module.spatial.parameters(), 'name': 'new-added'},
                    #                    {'params': self.net['feature_extractor'].module.bncbp.parameters(), 'name': 'new-added'},
                    #                    {'params': self.net['feature_extractor'].module.cbp_fc.parameters(), 'name': 'new-added'}
                ]

            self.optimizer_feature_extractor = torch.optim.SGD(params,
                                                               lr=self.opt.TRAIN.BASE_LR,
                                                               momentum=self.opt.TRAIN.MOMENTUM,
                                                               weight_decay=self.opt.TRAIN.WEIGHT_DECAY,
                                                               nesterov=True)

            self.optimizer_classifier = torch.optim.SGD([
                {'params': self.net['classifier'].parameters(), 'name': 'new-added'},
            ],
                lr=self.opt.TRAIN.BASE_LR,
                momentum=self.opt.TRAIN.MOMENTUM,
                weight_decay=self.opt.TRAIN.WEIGHT_DECAY,
                nesterov=True)
        else:
            raise NotImplementedError
        print('Optimizer built')

    def update_lr(self):
        if self.opt.TRAIN.LR_SCHEDULE == 'inv':
            if self.opt.TRAIN.PROCESS_COUNTER == 'epoch':
                lr = self.opt.TRAIN.BASE_LR / pow((1 + self.opt.INV.ALPHA * self.epoch / self.opt.TRAIN.MAX_EPOCH),
                                                  self.opt.INV.BETA)
            elif self.opt.TRAIN.PROCESS_COUNTER == 'iteration':
                lr = self.opt.TRAIN.BASE_LR / pow(
                    (1 + self.opt.INV.ALPHA * self.iters / (self.opt.TRAIN.MAX_EPOCH * self.iters_per_epoch)),
                    self.opt.INV.BETA)
            else:
                raise NotImplementedError
        elif self.opt.TRAIN.LR_SCHEDULE == 'fix':
            lr = self.opt.TRAIN.BASE_LR
        else:
            raise NotImplementedError
        lr_pretrain = lr * 0.1 if self.opt.TRAIN.LR_SCHEDULE != 'fix' else 0.0
        print(f'the lr is: {lr:.6f} and pretrain_lr is {lr_pretrain:.6f}')
        for param_group in self.optimizer_feature_extractor.param_groups:
            if param_group['name'] == 'pre-trained':
                param_group['lr'] = lr_pretrain
            elif param_group['name'] == 'new-added':
                param_group['lr'] = lr
            elif param_group['name'] == 'new-added-big':
                param_group['lr'] = lr * 10
            elif param_group['name'] == 'fixed':  ## Fix the lr as 0 can not fix the runing mean/var of the BN layer
                param_group['lr'] = 0

        for param_group in self.optimizer_classifier.param_groups:
            if param_group['name'] == 'pre-trained':
                param_group['lr'] = lr_pretrain
            elif param_group['name'] == 'new-added':
                param_group['lr'] = lr
            elif param_group['name'] == 'fixed':  ## Fix the lr as 0 can not fix the runing mean/var of the BN layer
                param_group['lr'] = 0

        return lr

    def save_ckpt(self, suffix=None):
        suffix = suffix if suffix is not None else f"{self.epoch:03d}"
        save_path = self.opt.SAVE_DIR
        ckpt_resume = os.path.join(save_path, f'ckpt_{suffix}.resume')
        pickle_ckpt_resume = os.path.join(save_path, f'ckpt_{suffix}_resume.pkl')
        torch.save({'epoch': self.epoch,
                    'best_prec1': self.best_prec1,
                    'feature_extractor_state_dict': self.net['feature_extractor'].state_dict(),
                    'classifier_state_dict': self.net['classifier'].state_dict()
                    }, ckpt_resume)
        with open(pickle_ckpt_resume, 'wb') as file:
            # A new file will be created
            pickle.dump(self.opt, file)

        print(f"Saving checkpoint at {ckpt_resume}")
