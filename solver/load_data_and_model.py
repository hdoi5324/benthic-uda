import torch

from config.config import cfg
from solver.generate_features import init_data
from utils.utils import print_networks


def load_data_and_model(args):
    ## Algorithm proposed in our CVPR19 paper: Domain-Symnetric Networks for Adversarial Domain Adaptation
    ## It is the same with our previous implementation of https://github.com/YBZh/SymNets
    if args.method == 'SymmNetsV1':
        if cfg.DATASET.DATASET == 'Digits':
            raise NotImplementedError
        else:
            from models.resnet_SymmNet import resnet as Model
            from data.prepare_data import generate_dataloader as Dataloader
        feature_extractor, classifier = Model()
        # feature_extractor = torch.nn.DataParallel(feature_extractor)
        # classifier = torch.nn.DataParallel(classifier)
        if torch.cuda.is_available():
            feature_extractor.cuda()
            classifier.cuda()
        net = {'feature_extractor': feature_extractor, 'classifier': classifier}
        print_networks(net['feature_extractor'], False)
        print_networks(net['classifier'], False)
    else:
        raise NotImplementedError("Currently don't support the specified method: %s." % (args.method))

    if cfg.RESUME != '':
        resume_dict = torch.load(cfg.RESUME)
        net['feature_extractor'].load_state_dict(resume_dict['feature_extractor_state_dict'])
        net['classifier'].load_state_dict(resume_dict['classifier_state_dict'])
        best_prec1 = resume_dict['best_prec1']
        epoch = resume_dict['epoch']
    dataloaders = Dataloader()
    train_data, test_data = init_data(dataloaders)
    return net, train_data, test_data
