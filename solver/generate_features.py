import torch
import numpy as np
from utils.utils import to_cuda


def init_data(dataloaders):
    train_data = {key: dict() for key in dataloaders if key != 'test'}
    for key in train_data.keys():
        if key not in dataloaders:
            continue
        cur_dataloader = dataloaders[key]
        train_data[key]['loader'] = cur_dataloader
        train_data[key]['iterator'] = None

    if 'test' in dataloaders:
        test_data = dict()
        test_data['loader'] = dataloaders['test']

    return train_data, test_data

def get_features(test_data, feature_extractor):
        feature_extractor.eval()
        feature_tests, targets = [], []

        for i, (input, target) in enumerate(test_data['loader']):
            input = to_cuda(input)
            with torch.no_grad():
                feature_test = feature_extractor(input)
                feature_tests.append(np.array(feature_test.cpu().detach()))
            targets += target.detach().tolist()

        features = np.concatenate(feature_tests)

        return features, targets
