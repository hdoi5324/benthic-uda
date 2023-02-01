import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from config.config import cfg
from solver.generate_features import init_data, get_features
from utils.class_color_dict import class_color_dict
from utils.set_random_seed import set_random_seed
from solver.load_data_and_model import load_data_and_model

def plot_tsne(args, logger=None):
    set_random_seed(cfg.SEED)
    # method-specific setting

    net, _, test_data = load_data_and_model(args)
    create_tsne_plot(logger, net, test_data)


def create_tsne_plot(logger, net, test_data):
    features, gt = get_features(test_data, net["feature_extractor"])
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
    logger["plots/tsne"].upload(plt.gcf())
    plt.savefig(filename, dpi=300,
                bbox_inches='tight',
                pad_inches=0.5)
    plt.show()



