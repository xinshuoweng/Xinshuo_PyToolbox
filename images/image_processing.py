# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

from check import *

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

def bgr2rgb_npimage(np_image, debug=True):
	'''
	this function transpose the channels of a numpy image from C x H x W to H x W x C
	'''
	if debug:
		assert isnparray(np_image), 'the input is not a numpy'
		assert np_image.ndim == 3 and np_image.shape[0] == 3, 'the input numpy image does not have a good dimension: {}'.format(np_image.shape)

	return np.transpose(np_image, (1, 2, 0)) 

def	unnormalize_npimage(np_image, debug=True):
	'''
	un-normalize a numpy image and scale it to [0, 255]
	'''

	if debug:
		assert isnparray(np_image), 'the input is not a numpy'
		assert np_image.ndim == 3 and np_image.shape[2] == 3, 'the input numpy image does not have a good dimension: {}'.format(np_image.shape)

	min_val = np.min(np_image)
	max_val = np.max(np_image)

	# print np_image


	np_image = np_image - min_val
	np_image = np_image / (max_val - min_val)

	np_image = np_image * 255.

	# print np.min(np_image)
	# print np.max(np_image)
	# time.sleep(5)

	np_image = np_image.astype('uint8')

	# print np_image

	if debug:
		assert np.min(np_image) == 0 and np.max(np_image) == 255, 'the value range is not right [%d, %d]' % (np.min(np_image), np.max(np_image))

	return np_image
