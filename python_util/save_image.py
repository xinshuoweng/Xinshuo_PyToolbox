import matplotlib.pyplot as plt


def save_image(img, save_path):
	width = img.shape[1]
	height = img.shape[0]
	dpi = 80
	figsize = width / float(dpi), height / float(dpi)
	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes([0, 0, 1, 1])
	ax.axis('off')
	ax.imshow(img, interpolation='nearest')
	ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
	fig.savefig(save_path, dpi=dpi, transparent=True)
	plt.close(fig)
