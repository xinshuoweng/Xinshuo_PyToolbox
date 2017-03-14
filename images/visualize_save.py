import matplotlib.pyplot as plt


def visualize_save(im, save_path, vis=False):
    assert (im.ndim == 3 and im.shape[2] == 3) or im.ndim == 2, 'The input image is not valid while visualizing'

    dpi = 80  
    width = im.shape[1]
    height = im.shape[0]
    figsize = width / float(dpi), height / float(dpi)

    # im = im[:, :, (2, 1, 0)]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im, interpolation='nearest')
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    fig.savefig(save_path, dpi=dpi, transparent=True)
    if vis:
        plt.show()
    plt.close(fig)