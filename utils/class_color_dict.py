import matplotlib as mpl
from matplotlib import cm
import seaborn as sns


def get_my_cmap(n):
    """Colours made from https://colorbrewer2.org/"""
    if n <= 4:
        colors = ['#33a02c', '#a6cee3', '#1f78b4', '#b2df8a']
    elif n <= 9:
        colors = (['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'])
    elif n <= 12:
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
                  '#6a3d9a', '#ffff99', '#b15928']
    else:
        colors = 'tab20'
    return sns.color_palette(colors)


def class_color_dict(num_classes, cmap=None, for_PIL=False):
    # Get a list of colours at least as long as num_classes
    if cmap is None:
        colors = get_my_cmap(num_classes)
    else:
        colors = sns.color_palette(cmap)

    # Use large color map if there aren't enough
    orig_colors = colors
    while len(colors) < num_classes:
        colors += orig_colors

    # Convert to dictionary which is more robust with PIL/Seaborn
    color_dict = {i: colors[i] for i in range(num_classes)}

    # Convert colours to integer tuples for PIL
    if for_PIL:
        orig_color_dict = color_dict
        for i in range(num_classes):
            c = orig_color_dict[i]
            color_dict[i] = tuple([int(j * 255) for j in c])

    return color_dict
