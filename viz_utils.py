import numpy as np

import matplotlib
import matplotlib.pyplot as plt


_COLOR_MAP_LOSS = np.array([
    [192, 192, 255],
    [ 96, 192,  96],
    [192, 255, 192],
    [192, 192, 192],
    [  0,   0,   0]  # nodata
]) / 255.0

_COLOR_MAP_GAIN = np.array([
    [  0,   0, 255],
    [  0, 128,   0],
    [128, 255, 128],
    [128, 128, 128],
    [  0,   0,   0]  # nodata
]) / 255.0

_COLOR_MAP_LC4 = np.array([
    [  0,   0, 255],
    [  0, 128,   0],
    [128, 255, 128],
    [128,  96,  96],
    [  0,   0,   0]  # nodata
]) / 255.0

_COLOR_MAP_NLCD16 = np.array([
    [  0,   0,   0], # nodata
    [ 70, 107, 159],
    [209, 222, 248],
    [222, 197, 197],
    [217, 146, 130],
    [235,   0,   0],
    [171,   0,   0],
    [179, 172, 159],
    [104, 171,  95],
    [ 28,  95,  44],
    [181, 197, 143],
    [204, 184, 121],
    [223, 223, 194],
    [220, 217,  57],
    [171, 108,  40],
    [184, 217, 235],
    [108, 159, 184],
]) / 255.0

CMAP_LOSS = matplotlib.colors.ListedColormap(_COLOR_MAP_LOSS)
CMAP_GAIN = matplotlib.colors.ListedColormap(_COLOR_MAP_GAIN)
CMAP_LC = matplotlib.colors.ListedColormap(_COLOR_MAP_LC4)
CMAP_NLCD = matplotlib.colors.ListedColormap(_COLOR_MAP_NLCD16)

def show_legend(patches, labels):
    fig = plt.figure(figsize=(4,2))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis("off")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.legend(patches, labels, loc='center', fontsize=17, frameon=False)
    plt.show()
    plt.close()

def show_loss_legend():
    labels = [
        'Water loss',
        'Tree canopy loss',
        'Low vegetation loss',
        'Impervious surface loss',
        'No change'
    ]
    patches = [
        matplotlib.patches.Patch(facecolor=CMAP_LOSS(i), edgecolor='k')
        for i in range(5)
    ]

    show_legend(patches, labels)

def show_gain_legend():
    labels = [
        'Water gain',
        'Tree canopy gain',
        'Low vegetation gain',
        'Impervious surface gain',
        'No change'
    ]
    patches = [
        matplotlib.patches.Patch(facecolor=CMAP_GAIN(i), edgecolor='k')
        for i in range(5)
    ]

    show_legend(patches, labels)

def show_img(img, cmap, vmin, vmax, title=None):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if cmap is not None:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    else:
        ax.imshow(img)
        
    if title is not None:
        plt.title(title, fontsize=20)
    plt.show()
    plt.close()

def show_loss(predictions, title=None):
    img = predictions.copy()
    zero_mask = img == 0
    img = img // 4
    img[zero_mask] = 4

    show_img(img, CMAP_LOSS, 0, len(_COLOR_MAP_LOSS) - 1, title)

def show_gain(predictions, title=None):
    img = predictions.copy()
    zero_mask = img == 0
    img = img % 4
    img[zero_mask] = 4

    show_img(img, CMAP_GAIN, 0, len(_COLOR_MAP_GAIN) - 1, title)