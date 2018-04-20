# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import math, cv2
import numpy as np
from PIL import Image

from private import safe_image, safe_image_like, safe_batch_deep_image, safe_batch_image
from xinshuo_math.python.private import safe_npdata, safe_angle, safe_ptsarray
from xinshuo_miscellaneous import isfloatimage, iscolorimage, isnparray, isnpimage_dimension, isuintimage, isgrayimage, ispilimage, islist, isinteger, islistofnonnegativeinteger, isfloatnparray, isuintnparray, isimsize, isscalar
from xinshuo_math import hist_equalization, clip_bboxes_TLWH, get_center_crop_bbox
from xinshuo_visualization import visualize_image

############################################# color transform #################################
def gray2rgb(input_image, with_color=True, cmap='jet', warning=True, debug=True):
	'''
	convert a grayscale image (1-channel) to a rgb image
		
	parameters:
		input_image:	an pil or numpy image
		with_color:		add false colormap

	output:
		rgb_image:		an uint8 rgb numpy image
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')

	if debug:
		assert isgrayimage(np_image), 'the input numpy image is not correct: {}'.format(np_image.shape)
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	if with_color:
		if cmap == 'jet':
			rgb_image = cv2.applyColorMap(np_image, cv2.COLORMAP_JET)
		else: assert False, 'cmap %s is not supported' % cmap
	else:
		rgb_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
	return rgb_image

def rgb2hsv(input_image, warning=True, debug=True):
	'''
	convert a rgb image to a hsv image using opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		hsv_image: 		an uint8 hsv numpy image
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	hsv_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
	return hsv_img

def rgb2hsv_v2(input_image, warning=True, debug=True):
	'''
	convert a rgb image to a hsv image, using PIL package, not compatible with opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		hsv_image: 		an uint8 hsv numpy image
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use PIL'

	pil_rgb_img = Image.fromarray(np_image)
	pil_hsv_img = pil_rgb_img.convert('HSV')
	hsv_img = np.array(pil_hsv_img)
	return hsv_img

def hsv2rgb(input_image, warning=True, debug=True):
	'''
	convert a hsv image to a rgb image using opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		rgb_img: 		an uint8 rgb numpy image
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	rgb_img = cv2.cvtColor(np_image, cv2.COLOR_HSV2RGB)
	return rgb_img

def rgb2lab(input_image, warning=True, debug=True):
	'''
	convert a rgb image to a lab image using opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		lab_image: 		an uint8 lab numpy image
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	lab_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
	return lab_img

def lab2rgb(input_image, warning=True, debug=True):
	'''
	convert a lab image to a rgb image using opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		rgb_img: 		an uint8 rgb numpy image
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	rgb_img = cv2.cvtColor(np_image, cv2.COLOR_LAB2RGB)
	return rgb_img

def image_hist_equalization(input_image, warning=True, debug=True):
	'''
	do histogram equalization for an image: could be a color image or gray image
	the color space used for histogram equalization is LAB

	parameters:
		input_image:		a pil or numpy image

	outputs:
		equalized_image:	an uint8 numpy image (rgb or gray)
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isuintimage(np_image):
		np_image = np_image.astype('float32') / 255.

	if debug: assert isfloatimage(np_image), 'the input image should be a float image'

	if iscolorimage(np_image):
		hsv_image = rgb2lab(np_image, warning=warning, debug=debug)
		input_data = hsv_image[:, :, 0]			# extract the value channel
		equalized_hsv_image = (hist_equalization(input_data, num_bins=256, debug=debug) * 255.).astype('uint8')
		hsv_image[:, :, 0] = equalized_hsv_image
		equalized_image = lab2rgb(hsv_image, warning=warning, debug=debug)
	elif isgrayimage(np_image):
		equalized_image = (hist_equalization(np_image, num_bins=256, debug=debug) * 255.).astype('uint8')
	else: assert False, 'the input image is neither a color image or a grayscale image'

	return equalized_image

def image_hist_equalization_hsv(input_image, debug=True):
	'''
	do histogram equalization for an image: could be a color image or gray image
	the color space used for histogram equalization is HSV
	
	parameters:
		input_image:		a pil or numpy image

	outputs:
		equalized_image:	an uint8 numpy image (rgb or gray)
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isuintimage(np_image):
		np_image = np_image.astype('float32') / 255.

	if debug: assert isfloatimage(np_image), 'the input image should be a float image'

	if iscolorimage(np_image):
		hsv_image = rgb2hsv(np_image, warning=warning, debug=debug)
		input_data = hsv_image[:, :, 2]			# extract the value channel
		equalized_hsv_image = (hist_equalization(input_data, num_bins=256, debug=debug) * 255.).astype('uint8')
		hsv_image[:, :, 2] = equalized_hsv_image
		equalized_image = hsv2rgb(hsv_image, warning=warning, debug=debug)
	elif isgrayimage(np_image):
		equalized_image = (hist_equalization(np_image, num_bins=256, debug=debug) * 255.).astype('uint8')
	else: assert False, 'the input image is neither a color image or a grayscale image'

	return equalized_image

def image_mean(input_image, warning=True, debug=True):
	'''
	this function computes the mean image over batch of images

	parameters:
		input_image: 		NHWC numpy image, uint8 or float32

	outputs:
		mean_image:			HWC numpy image, uint8	
	'''
	np_image, isnan = safe_batch_image(input_image, warning=warning, debug=debug)
	if isuintnparray(np_image):
		np_image = np_image.astype('float32') / 255.		
	else: assert isfloatnparray(np_image), 'the input image-like array should be either an uint8 or float32 array' 

	mean_image = (np.mean(np_image, axis=0) * 255.).astype('uint8')

	return mean_image

def image_draw_mask(input_image, input_image_mask, transparency=0.3, warning=True, debug=True):
	'''
	draw a mask on top of an image with certain transparency

	parameters: 
		input_image:			a pil or numpy image
		input_image_mask:		a pil or numpy image
		transparency:			transparency factor

	outputs:
		masked_image:			uint8 numpy image
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	np_image_mask, _ = safe_image(input_image_mask, warning=warning, debug=debug)
	if debug: 
		assert isscalar(transparency), 'the transparency should be a scalar'
		assert np_image.shape == np_image_mask.shape, 'the shape of mask should be equal to the shape of input image'
	if isfloatimage(np_image): np_image = (np_image * 255.).astype('uint8')
	if isfloatimage(np_image_mask): np_image_mask = (np_image_mask * 255.).astype('uint8')

	pil_image, pil_image_mask = Image.fromarray(np_image), Image.fromarray(np_image_mask)
	masked_image = np.array(Image.blend(pil_image, pil_image_mask, alpha=transparency))

	return masked_image

def	image_normalize(input_image, warning=True, debug=True):
	'''
	normalize an image to an uint8 with range of [0, 255]
	note that: the input might not be an image because the value range might be arbitrary

	parameters:
		input_image:		pil image or image-like array, color or gray, float or uint

	outputs:
		np_image:			numpy uint8 image, normalized to [0, 255]
	'''
	np_image, isnan = safe_image_like(input_image, warning=warning, debug=debug)
	if isuintnparray(np_image):
		np_image = np_image.astype('float32') / 255.		
	else:
		assert isfloatnparray(np_image), 'the input image-like array should be either an uint8 or float32 array' 

	min_val = np.min(np_image)
	max_val = np.max(np_image)
	if isnan: np_image.fill(0)
	elif min_val == max_val:								# all same
		if warning:
			print('the input image has the same value over all the pixels')
		np_image.fill(0)
	else:													# normal case
		np_image -= min_val
		np_image = ((np_image / (max_val - min_val)) * 255.).astype('uint8')
		if debug: assert np.min(np_image) == 0 and np.max(np_image) == 255, 'the value range is not right [%f, %f]' % (np.min(np_image), np.max(np_image))

	return np_image.astype('uint8')

def generate_gaussian_heatmap(input_pts, image_size, std, downsample_factor=1.0):
	'''
	generate a heatmap based on the input points array, create a 2-D gaussian with given std around each points provided
	the mask is generated by the occlusion from the point array: only occlusion with -1 will be masked out
		0 -> invisible points without location
		1 -> visible points with location
		-1 -> visible points without location, masked

	parameters:
		input_pts:			a list of 3 elements, a listoflist of 3 elements: e.g., [[1,2], [5,6], [0, 1]],
							a numpy array with shape or (3, N) or (3, )
		image_size:			a tuple or list of numpy array with 2 elements, representing (height, width)
		std:				the standard deviation used for gaussian distribution
		downsample_factor:	how many times to downsample from the given image size

	outputs:
		np_image:			numpy uint8 image, normalized to [0, 255]
	'''
	pts_array = safe_ptsarray_occlusion(input_pts, warning=warning, debug=debug)
	if debug:
		assert isscalar(std), 'the standard deviation should be a scalar'
		assert isimsize(image_size), 'the image size is not correct'
	
	if isinstance(sigma, numbers.Number):
		sigma = np.zeros((input_pts.shape[1])) + sigma
	assert isinstance(sigma, np.ndarray) and len(sigma.shape) == 1 and sigma.shape[0] == pts.shape[1], 'The shape of sigma : {}'.format(sigma.shape)

	offset = downsample / 2.0 - 0.5
	num_pts, threshold = pts.shape[1], 0.01

	if visiable is None: visiable = pts[2, :].astype('bool')
	assert visiable.shape[0] == num_points

	transformed_label = np.fromfunction( lambda y, x, pid : ((offset + x*downsample - pts[0,pid])**2 \
													+ (offset + y*downsample - pts[1,pid])**2) \
													/ -2.0 / sigma[pid] / sigma[pid],
													(height, width, num_points), dtype=int)

	mask_heatmap      = np.ones((1, 1, num_points+1), dtype='float32')
	mask_heatmap[0, 0, :num_points] = visiable
	mask_heatmap[0, 0, num_points]  = 1

	transformed_label = np.exp(transformed_label)
	transformed_label[ transformed_label < threshold ] = 0
	transformed_label[ transformed_label >         1 ] = 1
	transformed_label = transformed_label * mask_heatmap[:, :, :num_points]

	background_label  = 1 - np.amax(transformed_label, axis=2)
	background_label[ background_label < 0 ] = 0
	heatmap           = np.concatenate((transformed_label, np.expand_dims(background_label, axis=2)), axis=2).astype('float32')

	return heatmap*mask_heatmap, mask_heatmap

############################################# format transform #################################
def image_rgb2bgr(input_image, warning=True, debug=True):
	'''
	this function converts a rgb image to a bgr image

	parameters:
		input_image:	a pil or numpy rgb image

	outputs:
		np_image:		a numpy bgr image
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if debug: assert iscolorimage(np_image), 'the input image is not a color image'
	
	np_image = np_image[:, :, ::-1] 			# convert RGB to BGR
	return np_image

def image_bgr2rgb(input_image, warning=True, debug=True):
	'''
	this function converts a bgr image to a rgb image

	parameters:
		input_image:	a pil or numpy bgr image

	outputs:
		np_image:		a numpy rgb image
	'''
	return rgb2bgr(input_image, warning=warning, debug=debug)

def image_hwc2chw(input_image, warning=True, debug=True):
	'''
	this function transpose the channels of an image from HWC to CHW

	parameters:
		input_image:	a pil or numpy HWC image

	outputs:
		np_image:		a numpy CHW image
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if debug: assert np_image.ndim == 3 and np_image.shape[2] == 3, 'the input numpy image does not have a good dimension: {}'.format(np_image.shape)

	return np.transpose(np_image, (2, 0, 1)) 

def image_chw2hwc(input_image, warning=True, debug=True):
	'''
	this function transpose the channels of an image from CHW to HWC

	parameters:
		input_image:	a numpy CHW image

	outputs:
		np_image:		a numpy HWC image
	'''

	if debug: isnparray(input_image), 'the input image is not a numpy'
	np_image = input_image.copy()
	if debug: assert np_image.ndim == 3 and np_image.shape[0] == 3, 'the input numpy image does not have a good dimension: {}'.format(np_image.shape)

	return np.transpose(np_image, (1, 2, 0)) 

def preprocess_batch_deep_image(input_image, pixel_mean=None, pixel_std=None, rgb2bgr=True, warning=True, debug=True):
	'''
	this function preprocesses batch of images to a deep image, 
	1: from rgb to bgr
	2: normalize the batch image based on mean and std
	3: convert (N)HWC to NCHW

	parameters:
		input_image:			NHWC numpy color image, where C is 3, uint8 or float32
		pixel_mean:				a float32 numpy array mean over 3 channels with shape of (3, ) or (1, )
		pixel_std:				a float32 numpy array std over 3 channels with shape of (3, ) or (1, )
		rgb2bgr:				true if the input image is rgb format, such that the output is bgr image

	outputs:
		np_image: 				NCHW float32 color numpy image, bgr format
	'''
	np_image, isnan = safe_batch_image(input_image, warning=warning, debug=debug)
	if debug: assert np_image.shape[-1] == 3, 'the input image should be a color image'
	if isuintnparray(np_image):
		np_image = np_image.astype('float32') / 255.		
	else: assert isfloatnparray(np_image), 'the input image-like array should be either an uint8 or float32 array' 
	if rgb2bgr: np_image = np_image[:, :, :, [2, 1, 0]]                 	# from rgb to bgr, currently NHWC

	# normalize the numpy image data
	if pixel_mean is not None:
		pixel_mean = safe_npdata(pixel_mean, warning=warning, debug=debug)
		if debug: 
			assert pixel_mean.shape == (3, ) or pixel_mean.shape == (1, ), 'pixel mean is not correct'
			assert (pixel_mean <= 1.0).all() and (pixel_mean >= 0.0).all(), 'mean value should be in range [0, 1].'
		
		if pixel_mean.shape == (3, ): pixel_mean_reshape = np.reshape(pixel_mean, (1, 1, 1, 3))
		elif pixel_mean.shape == (1, ): pixel_mean_reshape = np.reshape(np.repeat(pixel_mean, 3), (1, 1, 1, 3))
		else: assert False, 'pixel mean is not correct'
		np_image -= pixel_mean_reshape

	if pixel_std is not None:
		pixel_std = safe_npdata(pixel_std, warning=warning, debug=debug)
		if debug: 
			assert pixel_std.shape == (3, ) or pixel_std.shape == (1, ), 'pixel std is not correct'
			assert (pixel_std <= 1.0).all() and (pixel_std >= 0.0).all(), 'std value should be in range [0, 1].'
		if pixel_std.shape == (3, ): pixel_std_reshape = np.reshape(pixel_std, (1, 1, 1, 3))
		elif pixel_std.shape == (1, ): pixel_std_reshape = np.reshape(np.repeat(pixel_std, 3), (1, 1, 1, 3))
		else: assert False, 'pixel std is not correct'
		np_image /= pixel_std_reshape

	np_image = np.transpose(np_image, (0, 3, 1, 2))         				# NHWC to NCHW

	return np_image

def unpreprocess_batch_deep_image(input_image, pixel_mean=None, pixel_std=None, bgr2rgb=True, warning=True, debug=True):
	'''
	this function unpreprocesses batch of deep images, which uses chw format instead of hwc format in general
	1: un-normalize the batch image based on mean and std
	2: convert NCHW to NHWC
	3. from bgr to rgb

	parameters:
		input_image:			NCHW / CHW float32 numpy array, where C is 3
		pixel_mean:				a float32 numpy array mean over 3 channels with shape of (3, ) or (1, )
		pixel_std:				a float32 numpy array std over 3 channels with shape of (3, ) or (1, )
		bgr2rgb:				true if the input image is bgr format, such that the output is rgb image

	outputs:
		np_image: 				NHWC uint8 color numpy image, rgb format
	'''
	np_image, isnan = safe_batch_deep_image(input_image, warning=warning, debug=debug)
	if debug: assert isfloatnparray(np_image), 'the input image-like array should be either an uint8 or float32 array' 

	if pixel_std is not None:
		pixel_std = safe_npdata(pixel_std, warning=warning, debug=debug)
		if debug: 
			assert pixel_std.shape == (3, ) or pixel_std.shape == (1, ), 'pixel std is not correct'
			assert (pixel_std <= 1.0).all() and (pixel_std >= 0.0).all(), 'std value should be in range [0, 1].'
		if pixel_std.shape == (3, ): pixel_std_reshape = np.reshape(pixel_std, (1, 3, 1, 1))
		elif pixel_std.shape == (1, ): pixel_std_reshape = np.reshape(np.repeat(pixel_std, 3), (1, 3, 1, 1))
		else: assert False, 'pixel std is not correct'
		np_image *= pixel_std_reshape

	if pixel_mean is not None:
		pixel_mean = safe_npdata(pixel_mean, warning=warning, debug=debug)
		if debug: 
			assert pixel_mean.shape == (3, ) or pixel_mean.shape == (1, ), 'pixel mean is not correct'
			assert (pixel_mean <= 1.0).all() and (pixel_mean >= 0.0).all(), 'mean value should be in range [0, 1].'
		if pixel_mean.shape == (3, ): pixel_mean_reshape = np.reshape(pixel_mean, (1, 3, 1, 1))
		elif pixel_mean.shape == (1, ): pixel_mean_reshape = np.reshape(np.repeat(pixel_mean, 3), (1, 3, 1, 1))
		else: assert False, 'pixel mean is not correct'
		np_image += pixel_mean_reshape

	np_image = np.transpose(np_image, (0, 2, 3, 1))         			# permute to [batch, height, weight, channel]	
	if bgr2rgb:	np_image = np_image[:, :, :, [2, 1, 0]]             	# from bgr to rgb for color image
	np_image = (np_image * 255.).astype('uint8')

	return np_image

############################################# 2D transformation #################################
def image_pad_around(input_image, pad_rect, pad_value=0, warning=True, debug=True):
	'''
	this function is to pad given value to an image in provided region, all images in this function are floating images
	
	parameters:
		input_image:	an pil or numpy image
	  	pad_rect:   	a list of 4 non-negative integers, describing how many pixels to pad. The order is [left, top, right, bottom]
	  	pad_value:  	an intger between [0, 255]

	outputs:
		img_padded:		an uint8 numpy image with padding
	'''
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image): np_image = (np_image * 255.).astype('uint8')
	if len(np_image.shape) == 2: np_image = np.expand_dims(np_image, axis=2)		# extend the third channel if the image is grayscale

	if debug:
		assert isuintimage(np_image), 'the input image is not an uint8 image'
		assert isinteger(pad_value) and pad_value >= 0 and pad_value <= 255, 'the pad value should be an integer within [0, 255]'
		assert islistofnonnegativeinteger(pad_rect) and len(pad_rect) == 4, 'the input pad rect is not a list of 4 non-negative integers'

	im_height, im_width, im_channel = np_image.shape[0], np_image.shape[1], np_image.shape[2]

	# calculate the padded size of image
	pad_left, pad_top, pad_right, pad_bottom = pad_rect[0], pad_rect[1], pad_rect[2], pad_rect[3]
	new_height  = im_height + pad_top + pad_bottom
	new_width   = im_width + pad_left + pad_right

	# padding
	img_padded = np.zeros([new_height, new_width, im_channel]).astype('uint8')
	img_padded.fill(pad_value)
	img_padded[pad_top : new_height - pad_bottom, pad_left : new_width - pad_right, :] = np_image
	if img_padded.shape[2] == 1: img_padded = img_padded[:, :, 0]

	return img_padded

def image_crop_center(input_image, center_rect, pad_value=0, warning=True, debug=True):
	'''
	crop the image around a specific center with padded value around the empty area
	when the crop width/height are even, the cropped image has 1 additional pixel towards left/up

	parameters:
		center_rect:	a list contains [center_x, center_y, (crop_width, crop_height)]
		pad_value:		scalar within [0, 255]

	outputs:
		img_cropped:			an uint8 numpy image
		crop_bbox:				numpy array with shape of (1, 4), user-attempted cropping bbox, might out of boundary
		crop_bbox_clipped:		numpy array with shape of (1, 4), clipped bbox within the boundary
	'''	
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image): np_image = (np_image * 255.).astype('uint8')
	if len(np_image.shape) == 2: np_image = np.expand_dims(np_image, axis=2)		# extend the third channel if the image is grayscale

	# center_rect and pad_value are checked in get_crop_bbox and pad_around functions
	if debug:
		assert isuintimage(np_image), 'the input image is not an uint8 image'
	im_height, im_width = np_image.shape[0], np_image.shape[1]
	
	# calculate crop rectangles
	crop_bbox = get_center_crop_bbox(center_rect, im_width, im_height, debug=debug)
	crop_bbox_clipped = clip_bboxes_TLWH(crop_bbox, im_width, im_height, debug=debug)
	x1, y1, x2, y2 = crop_bbox_clipped[0, 0], crop_bbox_clipped[0, 1], crop_bbox_clipped[0, 0] + crop_bbox_clipped[0, 2], crop_bbox_clipped[0, 1] + crop_bbox_clipped[0, 3]
	img_cropped = np_image[y1 : y2, x1 : x2, :]

	# if original image is not enough to cover the crop area, we pad value around outside after cropping
	xmin, ymin, xmax, ymax = crop_bbox[0, 0], crop_bbox[0, 1], crop_bbox[0, 0] + crop_bbox[0, 2], crop_bbox[0, 1] + crop_bbox[0, 3]
	if (xmin < 0 or ymin < 0 or xmax > im_width or ymax > im_height):
		pad_left    = max(0 - xmin, 0)
		pad_top     = max(0 - ymin, 0)
		pad_right   = max(xmax - im_width, 0)
		pad_bottom  = max(ymax - im_height, 0)
		pad_rect 	= [pad_left, pad_top, pad_right, pad_bottom]
		img_cropped = image_pad_around(img_cropped, pad_rect=pad_rect, pad_value=pad_value, debug=debug)
	if len(img_cropped.shape) == 3 and img_cropped.shape[2] == 1: img_cropped = img_cropped[:, :, 0]

	return img_cropped, crop_bbox, crop_bbox_clipped

def image_resize(input_image, resize_factor=None, target_size=None, interp='bicubic', warning=True, debug=True):
	'''
	resize the image given a resize factor (e.g., 0.25), or given a target size (height, width)
	e.g., the input image has 600 x 800:
		1. given a resize factor of 0.25 -> results in an image with 150 x 200
		2. given a target size of (300, 400) -> results in an image with 300 x 400
	note that:
		resize_factor and target_size cannot exist at the same time

	parameters:
		input_image:		an pil or numpy image
		resize_factor:		a scalar
		target_size:		a list of tuple or numpy array with 2 elements, representing height and width
		interp:				interpolation methods: bicubic or bilinear

	outputs:
		resized_image:		a numpy uint8 image
	'''	
	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image): np_image = (np_image * 255.).astype('uint8')

	if debug:
		assert interp in ['bicubic', 'bilinear'], 'the interpolation method is not correct'
		assert (resize_factor is not None and target_size is None) or (resize_factor is None and target_size is not None), 'resize_factor and target_size cannot co-exist'

	if target_size is not None:
		if debug: assert isimsize(target_size), 'the input target size is not correct'
		target_width, target_height = int(round(target_size[1])), int(round(target_size[0]))
	elif resize_factor is not None:
		if debug: assert isscalar(resize_factor), 'the resize factor is not a scalar'
		height, width = np_image.shape[:2]
		target_width, target_height = int(round(resize_factor * width)), int(round(resize_factor * height))
	else: assert False, 'the target_size and resize_factor do not exist'

	if interp == 'bicubic':
	    resized_image = cv2.resize(np_image, (target_width, target_height), interpolation = cv2.INTER_CUBIC)
	elif interp == 'bilinear':
		resized_image = cv2.resize(np_image, (target_width, target_height), interpolation = cv2.INTER_LINEAR)
	else: assert False, 'interpolation is wrong'

	return resized_image

def image_rotate(input_image, input_angle, warning=True, debug=True):
	'''
	rotate the image given an angle in degree (e.g., 90). The rotation is counter-clockwise
	
	parameters:
		input_image:		an pil or numpy image
		input_angle:		a scalar

	outputs:
		rotated_image:		a numpy uint8 image
	'''	
	if debug: assert isscalar(input_angle), 'the input angle is not a scalar'
	rotation_angle = safe_angle(input_angle, warning=warning, debug=True)             # ensure to be in [-180, 180]

	np_image, _ = safe_image(input_image, warning=warning, debug=debug)
	if isfloatimage(np_image): np_image = (np_image * 255.).astype('uint8')
	pil_image = Image.fromarray(np_image)
	pil_image = pil_image.rotate(rotation_angle, expand=True)
	rotated_image = np.array(pil_image).astype('uint8')

	return rotated_image

def image_concatenate(image_list, im_size=[1600, 2560], grid_size=None, edge_factor=0.99, debug=True):
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

def hstack_images( images, gap ):
	images = [np.array(image) for image in images]
	imagelist = []
	for image in images:
		gap_shape = list(image.shape)
		gap_shape[1] = gap
		imagelist.append(image)
		imagelist.append(np.zeros(gap_shape).astype('uint8'))
	imagelist = imagelist[:-1]
	hstack = np.hstack( imagelist )
	return Image.fromarray( hstack )

def vstack_images( images, gap ):
	images = [np.array(image) for image in images]
	imagelist = []
	for image in images:
		gap_shape = list(image.shape)
		gap_shape[0] = gap
		imagelist.append(image)
		imagelist.append(np.zeros(gap_shape).astype('uint8'))
	imagelist = imagelist[:-1]
	hstack = np.vstack( imagelist )
	return Image.fromarray( hstack )


# centroid
# def find_tensor_peak_batch(heatmap, radius, downsample, threshold = 0.000001):
#   assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
#   assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
#   num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
#   # find the approximate location:
#   score, index = torch.max(heatmap.view(num_pts, -1), 1)
#   index_w = (index % W).float()
#   index_h = (index / W).float()
  
#   def normalize(x, L):
#     return -1. + 2. * x.data / (L-1)
#   boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
#   boxes[0] = normalize(boxes[0], W)
#   boxes[1] = normalize(boxes[1], H)
#   boxes[2] = normalize(boxes[2], W)
#   boxes[3] = normalize(boxes[3], H)
#   affine_parameter = torch.zeros((num_pts, 2, 3))
#   affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
#   affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
#   affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
#   affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
  
#   # extract the sub-region heatmap
#   theta = MU.np2variable(affine_parameter,heatmap.is_cuda,False)
#   grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
#   grid = F.affine_grid(theta, grid_size)
#   sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
#   sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

#   # slow for speed improvement
#   X = MU.np2variable(torch.arange(-radius, radius+1),heatmap.is_cuda,False).view(1, 1, radius*2+1)
#   Y = MU.np2variable(torch.arange(-radius, radius+1),heatmap.is_cuda,False).view(1, radius*2+1, 1)
  
#   sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
#   x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
#   y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h
     
#   x = x * downsample + downsample / 2.0 - 0.5
#   y = y * downsample + downsample / 2.0 - 0.5
#   return torch.stack([x, y],1), score

# TODO
def find_peaks(heatmap, thre):
    #filter = fspecial('gaussian', [3 3], 2)
    #map_smooth = conv2(map, filter, 'same')
    
    # variable initialization    

    map_smooth = np.array(heatmap)
    map_smooth[map_smooth < thre] = 0.0


    map_aug = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug1 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug2 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug3 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug4 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    
    # shift in different directions to find peak, only works for convex blob
    map_aug[1:-1, 1:-1] = map_smooth
    map_aug1[1:-1, 0:-2] = map_smooth
    map_aug2[1:-1, 2:] = map_smooth
    map_aug3[0:-2, 1:-1] = map_smooth
    map_aug4[2:, 2:] = map_smooth

    peakMap = np.multiply(np.multiply(np.multiply((map_aug > map_aug1),(map_aug > map_aug2)),(map_aug > map_aug3)),(map_aug > map_aug4))
    peakMap = peakMap[1:-1, 1:-1]

    idx_tuple = np.nonzero(peakMap)     # find 1
    Y = idx_tuple[0]
    X = idx_tuple[1]

    score = np.zeros([len(Y),1])
    for i in range(len(Y)):
        score[i] = heatmap[Y[i], X[i]]

    return X, Y, score


# def find_tensor_peak(heatmap, radius, downsample):
#   assert heatmap.dim() == 2, 'The dimension of the heatmap is wrong : {}'.format(heatmap.dim())
#   assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
#   H, W = heatmap.size(0), heatmap.size(1)
#   # find the approximate location:
#   score, index = torch.max(heatmap.view(-1), 0)
#   index = int(MU.variable2np(index))
#   index_h, index_w = np.unravel_index(index, (H,W))

#   sw, sh = int(index_w - radius),     int(index_h - radius)
#   ew, eh = int(index_w + radius + 1), int(index_h + radius + 1)
#   sw, sh = max(0, sw), max(0, sh)
#   ew, eh = min(W, ew), min(H, eh)
  
#   subregion = heatmap[sh:eh, sw:ew]
#   threshold = 0.000001
#   eps = np.finfo(float).eps

#   with torch.cuda.device_of(subregion):
#     X = MU.np2variable(torch.arange(sw, ew).unsqueeze(0))
#     Y = MU.np2variable(torch.arange(sh, eh).unsqueeze(1))

#   indicator = (subregion > threshold).type( type(subregion.data) )
#   eps = (subregion <= threshold).type( type(subregion.data) ) * eps
#   subregion = subregion * indicator + eps
  
#   x = torch.sum( subregion * X ) / torch.sum( subregion )
#   y = torch.sum( subregion * Y ) / torch.sum( subregion )
     
#   ## calculate the score
#   np_x, np_y = MU.variable2np(x), MU.variable2np(y)
#   x2, y2 = min(W-1, int(np.ceil(np_x))), min(H-1, int(np.ceil(np_y)))
#   x1, y1 = max(0, x2-1), max(0, y2-1)
#   ## Bilinear interpolation
#   if x1 == x2: 
#     R1, R2 = heatmap[y1, x1], heatmap[y1, x2]
#   else:
#     R1 = (x2-x)/(x2-x1)*heatmap[y1, x1] + (x-x1)/(x2-x1)*heatmap[y1, x2]
#     R2 = (x2-x)/(x2-x1)*heatmap[y2, x1] + (x-x1)/(x2-x1)*heatmap[y2, x2]
#   if y1 == y2:
#     score = R1
#   else:
#     score = (y2-y)/(y2-y1)*R1 + (y-y1)/(y2-y1)*R2
     
#   x = x * downsample + downsample / 2.0 - 0.5
#   y = y * downsample + downsample / 2.0 - 0.5
#   return torch.cat([x, y]), score
