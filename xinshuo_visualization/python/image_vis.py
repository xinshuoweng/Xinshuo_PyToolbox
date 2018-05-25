# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions for visualizing on images 
from private import save_vis_close_helper, get_fig_ax_helper
from geometry_vis import visualize_pts_array, visualize_bbox
from xinshuo_images.python.private import safe_image
from xinshuo_images import image_bgr2rgb
from xinshuo_miscellaneous import  remove_list_from_list, isdict, islist, iscolorimage_dimension, isgrayimage_dimension

def visualize_image(input_image, bgr2rgb=False, save_path=None, vis=False, warning=True, debug=True, closefig=True):
	'''
	visualize an image

	parameters:
		input_image:		a pil or numpy image
		bgr2rgb:			true if the image needs to be converted from bgr to rgb
		save_path:			a path to save. Do not save if it is None
		closefig:			False if you want to add more elements on the image

	outputs:
		fig, ax:			figure handle for future use
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	width, height = np_image.shape[1], np_image.shape[0]
	fig, _ = get_fig_ax_helper(fig=None, ax=None, width=width, height=height)
	ax = fig.add_axes([0, 0, 1, 1])
	ax.axis('off')

	# display image
	if iscolorimage_dimension(np_image):
		if bgr2rgb: np_image = image_bgr2rgb(np_image)
		ax.imshow(np_image, interpolation='nearest')
	elif isgrayimage_dimension(np_image):
		np_image = np_image.reshape(np_image.shape[0], np_image.shape[1])
		ax.imshow(np_image, interpolation='nearest', cmap='gray')
	else: assert False, 'unknown image type'

	ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
	return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, warning=warning, closefig=closefig)

def visualize_image_with_pts(input_image, input_pts, color_index=0, pts_size=20, vis_threshold=0.3, label=False, label_list=None, label_size=20, 
	bgr2rgb=False, save_path=None, vis=False, warning=True, debug=True, closefig=True):
	'''
	visualize an image and plot points on top of it

	parameters:
		input_image:	a pil or numpy image
		input_pts:		2(3) x num_pts numpy array or a dictionary of 2(3) x num_pts array
						when there are 3 channels in pts, the third one denotes the occlusion/confidence flag		
						occlusion: 0 -> invisible and not annotated, 1 -> visible and annotated, -1 -> visible but not annotated
		color_index:	a scalar or a list of color indexes
		vis_threshold:	the points with confidence above the threshold will be drawn
		label:			determine to add text label for each point, if label list is None, then an automatic list is created
		label_list:		label string for all points, if label list is not None, the label is True automatically
						if the input points is a dictionary, then every point array in the dict follow the same label list
		bgr2rgb:		true if the image needs to be converted from bgr to rgb

	outputs:
		fig, ax:		figure handle for future use
	'''
	fig, ax = visualize_image(input_image, bgr2rgb=bgr2rgb, vis=False, save_path=None, warning=warning, debug=debug, closefig=False)
	if isdict(input_pts):
		for pts_id, pts_array_tmp in input_pts.items():
			visualize_pts_array(pts_array_tmp, fig=fig, ax=ax, color_index=color_index, pts_size=pts_size, label=label, label_list=label_list, label_size=label_size, 
				plot_occl=False, covariance=False, xlim=None, ylim=None, vis_threshold=vis_threshold, debug=debug, vis=False, save_path=None, warning=warning, closefig=False)
			color_index += 1
	else: visualize_pts_array(input_pts, fig=fig, ax=ax, color_index=color_index, pts_size=pts_size, label=label, label_list=label_list, label_size=label_size, 
		plot_occl=False, covariance=False, xlim=None, ylim=None, vis_threshold=vis_threshold, debug=debug, vis=False, save_path=None, warning=warning, closefig=False)
	return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, warning=warning, closefig=closefig)

def visualize_image_with_bbox(input_image, input_bbox, linewidth=0.5, edge_color_index=20,
	bgr2rgb=False, save_path=None, vis=False, warning=True, debug=True, closefig=True):
	'''
	visualize image and plot bounding boxes on top of it

	parameter:
		input_image:	a pil or numpy image
		input_bbox:		a list of 4 elements, a listoflist of 4 elements: e.g., [[1,2,3,4], [5,6,7,8]],
						a numpy array with shape or (N, 4) or (4, )
						TLBR format

	outputs:
		fig, ax:		figure handle for future use
	'''
	fig, ax = visualize_image(input_image, bgr2rgb=bgr2rgb, vis=False, save_path=None, warning=warning, debug=debug, closefig=False)
	fig, ax = visualize_bbox(input_bbox, linewidth=linewidth, edge_color_index=edge_color_index, fig=fig, ax=ax, debug=debug, vis=False, save_path=None, warning=warning, closefig=False)
	return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, warning=warning, closefig=closefig)

def visualize_image_with_pts_bbox(image, pts_array, window_size, pts_size=20, label=False, label_list=None, color_index=0, 
	bgr2rgb=False, save_path=None, vis=False, warning=True, debug=True, closefig=True):
	'''
	plot a set of points on top of an image with bbox around all points

	parameters
		pts_array:              2 x N
	'''
	fig, ax = visualize_image_with_pts(image, pts_array, pts_size=pts_size, label=label, label_list=label_list, color_index=color_index, bgr2rgb=bgr2rgb, debug=debug, vis=False, save_path=None, warning=warning, closefig=False)
	num_pts = pts_array.shape[1]
	# center_bbox = 
	input_bbox = get_center_crop_bbox(pts_array, window_size, window_size, warning=warning, debug=debug)
	fig, ax = visualize_bbox(input_bbox, linewidth=linewidth, edge_color_index=edge_color_index, fig=fig, ax=ax, debug=debug, vis=False, save_path=None, warning=warning, closefig=False)
	return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, warning=warning, closefig=closefig)