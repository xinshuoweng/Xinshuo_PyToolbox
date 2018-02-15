# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import math
import numpy as np
from PIL import Image

from xinshuo_python import *

def imagecoor2cartesian(pts, debug=True):
	'''
	change the coordinate system from image coordinate system to normal cartesian system, basically reverse the y coordinate

	parameter: 
		pts:	a single point (list, tuple, numpy array) or a 2 x N numpy array representing a set of points

	return:
		pts:	a tuple if only single point comes in or a 2 x N numpy array
	'''
	return cartesian2imagecoor(pts, debug=debug)

def cartesian2imagecoor(pts, debug=True):
	'''
	change the coordinate system from normal cartesian system back to image coordinate system, basically reverse the y coordinate
	
	parameter: 
		pts:	a single point (list, tuple, numpy array) or a 2 x N numpy array representing a set of points

	return:
		pts:	a tuple if only single point comes in or a 2 x N numpy array
	'''
	if debug:
		assert is2dpts(pts) or (isnparray(pts) and pts.shape[0] == 2 and pts.shape[1] > 0), 'point is not correct'
	
	if is2dpts(pts):
		if isnparray(pts):
			pts = np.reshape(pts, (2, ))
		return (pts[0], -pts[1])
	else:
		pts[1, :] = -pts[1, :]
		return pts

def imagecoor2cartesian_center(image_shape, debug=True):
	'''
	given an image shape, return 2 functions which change the original image coordinate to centered cartesian coordinate system
	basically the origin is in the center of the image
	
	for example:
		if the image shape is (480, 640) and the top left point is (0, 0), after passing throught forward_map function, it returns (-320, 240)
		for the bottom right point (639, 479), it returns (319, -239)
	'''
	if debug:
		assert (istuple(image_shape) or islist(image_shape) or isnparray(image_shape)) and np.array(image_shape).size == 2, 'input image shape is not correct'

	width = image_shape[1]
	height = image_shape[0]

	def forward_map(pts, debug=True):
		if debug:
			assert is2dpts(pts), 'input 2d point is not correct'
			assert pts[0] >= 0 and pts[0] < width and isinteger(pts[0]), 'x coordinate is out of range %d should in [%d, %d)' % (pts[0], 0, width)
			assert pts[1] >= 0 and pts[1] < height and isinteger(pts[1]), 'y coordinate is out of range %d shoud in [%d, %d)' % (pts[1], 0, height)

		car_pts = imagecoor2cartesian(pts, debug=debug)
		car_pts = np.array(car_pts)
		car_pts[0] += -width/2		# shift x axis half length of width to the right
		car_pts[1] += height/2		# shigt y axis hald length of height downside
		return (car_pts[0], car_pts[1])

	def backward_map(pts, debug=True):
		if debug:
			assert is2dpts(pts), 'input 2d point is not correct'
			assert is2dpts(pts), 'input 2d point is not correct'
			assert pts[0] >= -width/2 and pts[0] < width/2 and isinteger(pts[0]), 'x coordinate is out of range %d should in [%d, %d)' % (pts[0], -width/2, width/2)
			assert pts[1] > -height/2 and pts[1] <= height/2 and isinteger(pts[1]), 'y coordinate is out of range %d shoud in (%d, %d]' % (pts[1], -height/2, height/2)

		pts = np.array(pts)
		pts[0] += width/2		
		pts[1] += -height/2		
		img_pts = cartesian2imagecoor(pts, debug=debug)
		return img_pts
		
	return forward_map, backward_map

def generate_mean_image(images_dir, save_path, debug=True, vis=False):
	'''
	this function generates the mean image over all images. It assume the image has the same size

	parameters:
			images_dir: 		path to all images
	'''
	if debug:
		assert is_path_exists(images_dir), 'the image path is not existing at %s' % images_dir
		assert is_path_exists_or_creable(save_path), 'the path for saving is not correct: %s' % save_path

	print('loading image data from %s' % images_dir)
	imagelist, num_images = load_list_from_folders(images_dir, ext_filter=['png', 'jpg', 'jpeg'], depth=None)
	print('{} images loaded'.format(num_images))

	# load the first image to see the image size
	img = Image.open(imagelist[0]).convert('L')
	width, height = img.size

	image_blob = np.zeros((height, width, num_images), dtype='float32')
	count = 0
	for image_path in imagelist:
		print('generating sharpness: processing %d/%d' % (count+1, num_images))
		img = Image.open(image_path).convert('L')	
		image_blob[:, :, count] = np.asarray(img, dtype='float32') / 255
		count += 1

	mean_im = np.mean(image_blob, axis=2)
	visualize_image(mean_im, debug=debug, vis=vis, save=True, save_path=save_path)

def pil2cv_colorimage(pil_image, debug=True, vis=False):
	'''
	this function converts a PIL image to a cv2 image, which has RGB and BGR format respectively
	'''
	if debug:
		assert ispilimage(pil_image), 'the input image is not a PIL image'

	cv_image = np.array(pil_image)
	cv_image = cv_image[:, :, ::-1].copy() 			# convert RGB to BGR

	return cv_image

def chw2hwc_npimage(np_image, debug=True):
	'''
	this function transpose the channels of a numpy image from C x H x W to H x W x C
	'''
	if debug:
		assert isnparray(np_image), 'the input is not a numpy'
		assert np_image.ndim == 3 and np_image.shape[0] == 3, 'the input numpy image does not have a good dimension: {}'.format(np_image.shape)

	return np.transpose(np_image, (1, 2, 0)) 

def	unnormalize_npimage(np_image, debug=True):
	'''
	un-normalize a numpy image and scale it to [0, 255], uint8
	'''
	if debug:
		assert isnpimage_dimension(np_image), 'the input numpy image is not correct: {}'.format(np_image.shape)

	min_val = np.min(np_image)
	max_val = np.max(np_image)
	if math.isnan(min_val) or math.isnan(max_val):			# with nan
		assert False, 'the input image has nan'
	elif min_val == max_val:								# all same
		np_image.fill(0)
	else:													# normal case
		np_image = np_image - min_val
		np_image = np_image / (max_val - min_val)
		np_image = np_image * 255.
		if debug:
			assert np.min(np_image) == 0 and np.max(np_image) == 255, 'the value range is not right [%f, %f]' % (min_val, max_val)

	np_image = np_image.astype('uint8')

	return np_image

def concatenate_grid(image_list, im_size=[1600, 2560], grid_size=None, edge_factor=0.99, debug=True):
	'''
	concatenate a list of images automatically

	parameters:	
		image_list: 		a list of numpy array
		im_size:			a tuple or list of numpy array for [H, W]
		edge_factor:		the margin between images after concatenation, bigger, the edge is smaller, [0, 1]
	'''
	if debug:
		assert islist(image_list) and all(ispilimage(image_tmp) for image_tmp in image_list), 'the input is not a list of image'
		assert issize(im_size), 'the input image size is not correct'
		if grid_size is not None:
			assert issize(grid_size), 'the input grid size is not correct'

	num_images = len(image_list)

	if grid_size is None:
		num_rows = int(np.sqrt(num_images))
		num_cols = int(np.ceil(num_images * 1.0 / num_rows))
	else:
		num_rows = grid_size[0]
		num_cols = grid_size[1]

	window_height = im_size[0]
	window_width = im_size[1]
	
	grid_height = int(window_height / num_rows)
	grid_width  = int(window_width  / num_cols)
	im_height   = int(grid_height   * edge_factor)
	im_width 	= int(grid_width 	 * edge_factor)
	im_channel 	= 3

	# concatenate
	image_merged = np.zeros((window_height, window_width, im_channel), dtype='uint8')
	for image_index in range(num_images):
		image_tmp = image_list[image_index]
		image_tmp = image_tmp.resize((im_width, im_height), Image.ANTIALIAS)
		image_tmp = image_tmp.convert('RGB')

		rows_index = int(np.ceil((image_index+1.0) / num_cols))			# 1-indexed
		cols_index = image_index+1 - (rows_index - 1) * num_cols	# 1-indexed
		rows_start = 1 + (rows_index - 1) * grid_height				# 1-indexed
		rows_end   = rows_start + im_height							# 1-indexed
		cols_start = 1 + (cols_index - 1) * grid_width				# 1-indexed
		cols_end   = cols_start + im_width							# 1-indexed

		image_merged[rows_start:rows_end, cols_start : cols_end, :] = np.array(image_tmp)

	return image_merged




# this function is to pad given value to an image in provided region, all images in this function are floating images
# parameters:
#   img:        a floating image
#   pad_rect:   4 element array, which describes where to pad the value. The order is [left, top, right, bottom]
#   pad_value:  a scalar defines what value we should pad
def pad_around(img, pad_rect, pad_value):

    [im_height, im_width, channel] = img.shape
    
    # calculate the new size of image
    pad_left    = pad_rect[0];
    pad_top     = pad_rect[1];
    pad_right   = pad_rect[2];
    pad_bottom  = pad_rect[3];
    new_height  = im_height + pad_top + pad_bottom;
    new_width   = im_width + pad_left + pad_right;
    
    # pad
    padded = np.zeros([new_height, new_width, channel]);
    padded[:] = pad_value;
    padded[pad_top: new_height-pad_bottom, pad_left: new_width-pad_right, :] = img;
    return padded

def crop_center(img1, rect, pad_value):
# rect is XYWH

    if not pad_value:
        pad_value = 128

    # calculate crop rectangles
    [im_height, im_width, im_channel] = img1.shape
    im_size = [im_height, im_width]

    rect = [int(x) for x in rect]
    if len(rect) == 4:            # crop around the given center and width and height
        center_x = rect[0]
        center_y = rect[1]
        crop_width = rect[2]
        crop_height = rect[3]
    else:                            # crop around the center of the image
        center_x = math.ceil(im_width/2)
        center_y = math.ceil(im_height/2)   
        crop_width = rect[0]
        crop_height = rect[1]
    
    # calculate cropped region
    xmin = int(center_x - math.ceil(crop_width/2) + 1) -1
    ymin = int(center_y - math.ceil(crop_height/2) + 1) -1
    xmax = int(xmin + crop_width - 1)
    ymax = int(ymin + crop_height - 1)
    
    crop_rect = [xmin, ymin, crop_width - 1, crop_height - 1]
    tmp_min_x = xmin if xmin>=0 else 0
    tmp_max_x = xmax if xmax<img1.shape[1] else (img1.shape[1]-1)
    tmp_min_y = ymin if ymin>=0 else 0
    tmp_max_y = ymax if ymax<img1.shape[0] else (img1.shape[0]-1)
    cropped = img1[tmp_min_y:tmp_max_y+1, tmp_min_x:tmp_max_x+1, :]

    # if original image is not enough to cover the crop area, we pad value around outside after cropping
    if (xmin < 0 or ymin < 0 or xmax > im_width-1 or ymax > im_height-1):
        pad_left    = max(0 - xmin, 0)
        pad_top     = max(0 - ymin, 0)
        pad_right   = max(xmax - (im_width-1), 0)
        pad_bottom  = max(ymax - (im_height-1), 0)
        if pad_left > 0:
            tmp = np.ones((cropped.shape[0], pad_left + cropped.shape[1], img1.shape[2]))*pad_value
            tmp[: , pad_left:, :] = cropped
            cropped = tmp
        if pad_right > 0:
            tmp = np.ones((cropped.shape[0], pad_right + cropped.shape[1], img1.shape[2]))*pad_value
            tmp[:, :cropped.shape[1], :] = cropped
            cropped = tmp
        if pad_top > 0:
            tmp = np.ones((pad_top + cropped.shape[0], cropped.shape[1], img1.shape[2])) * pad_value
            tmp[pad_top:, :, :] = cropped
            cropped = tmp
        if pad_bottom > 0:
            tmp = np.ones((pad_bottom + cropped.shape[0], cropped.shape[1], img1.shape[2])) * pad_value
            tmp[:cropped.shape[0] , :, :] = cropped
            cropped = tmp

    # TODO: with padding
    [im_height, im_width, im_channel] = img1.shape
    
    crop_rect_ori = toolbox.bbox.clip_bboxes_TLWH(crop_rect, im_width, im_height)
    return cropped, crop_rect, crop_rect_ori
