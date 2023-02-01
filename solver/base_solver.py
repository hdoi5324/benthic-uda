import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from config.config import cfg
from utils.class_color_dict import class_color_dict
from utils.utils import to_cuda


class BaseSolver:
    def __init__(self, net, dataloaders, logger=None, **kwargs):
        self.opt = cfg
        self.net = net
        self.dataloaders = dataloaders
        self.logger = logger
        self.CELoss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.CELoss.cuda()
        self.epoch = 0
        self.iters = 0
        self.best_prec1 = 0
        self.iters_per_epoch = None
        self.build_optimizer()
        self.init_data(self.dataloaders)

    def init_data(self, dataloaders):
        self.train_data = {key: dict() for key in dataloaders if key != 'test'}
        for key in self.train_data.keys():
            if key not in dataloaders:
                continue
            cur_dataloader = dataloaders[key]
            self.train_data[key]['loader'] = cur_dataloader
            self.train_data[key]['iterator'] = None

        if 'test' in dataloaders:
            self.test_data = dict()
            self.test_data['loader'] = dataloaders['test']

    def build_optimizer(self):
        print('Optimizer built')

    def complete_training(self):
        if self.epoch >= self.opt.TRAIN.MAX_EPOCH:
            return True

    def solve(self):
        print('Training Done!')

    def get_samples(self, data_name):
        assert (data_name in self.train_data)
        assert ('loader' in self.train_data[data_name] and \
                'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample

    def update_network(self, **kwargs):
        pass

    def create_tsne_plot(self):
        features, gt = self.get_features()
        tsne = TSNE(perplexity=30, early_exaggeration=125, init='pca')
        embedding = tsne.fit_transform(features)
        results_df = pd.DataFrame(embedding, columns=['x', 'y'])
        results_df["gt"] = gt
        results_df["domain"] = "TunaSand"
        # define figure size
        filename = f"tsne_{cfg.DATASET.SOURCE_NAME}"
        num_classes = cfg.DATASET.NUM_CLASSES
        color_dict = class_color_dict(num_classes)
        sns.set(rc={"figure.figsize": (20, 12)})  # width, height
        sns.set_theme(style="white")
        sns.relplot(x="x", y="y", data=results_df, hue="gt", style="domain", palette=color_dict)  # row="domain")
        sns.despine(left=True, bottom=True)
        self.logger["plots/tsne"].upload(plt.gcf())
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.5)

    def get_features(self):
        self.net["feature_extractor"].eval()
        feature_tests, targets = [], []

        for i, (input, target) in enumerate(self.test_data['loader']):
            input = to_cuda(input)
            with torch.no_grad():
                feature_test = self.net["feature_extractor"](input)
                feature_tests.append(np.array(feature_test.cpu().detach()))
            targets += target.detach().tolist()

        features = np.concatenate(feature_tests)

        return features, targets
