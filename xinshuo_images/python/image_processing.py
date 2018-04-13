# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import math, cv2
import numpy as np
from PIL import Image

from private import safe_image
from xinshuo_miscellaneous import isfloatimage, iscolorimage, isnparray, isnpimage_dimension, isuintimage, isgrayimage, ispilimage, islist, isinteger, islistofnonnegativeinteger
from xinshuo_math import hist_equalization, clip_bboxes_TLWH, get_center_crop_bbox
from xinshuo_visualization import visualize_image

############################################# color transform #################################
def gray2rgb(input_image, with_color=True, cmap='jet', debug=True):
	'''
	convert a grayscale image (1-channel) to a rgb image
		
	parameters:
		input_image:	an pil or numpy image
		with_color:		add false colormap

	output:
		rgb_image:		an uint8 rgb numpy image
	'''
	np_image = safe_image(input_image)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')

	if debug:
		assert isgrayimage(np_image), 'the input numpy image is not correct: {}'.format(np_image.shape)
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	if with_color:
		if cmap == 'jet':
			rgb_image = cv2.applyColorMap(np_image, cv2.COLORMAP_JET)
		else:
			assert False, 'cmap %s is not supported' % cmap
	else:
		rgb_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
	return rgb_image

def rgb2hsv(input_image, debug=True):
	'''
	convert a rgb image to a hsv image using opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		hsv_image: 		an uint8 hsv numpy image
	'''
	np_image = safe_image(input_image)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	hsv_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
	return hsv_img

def rgb2hsv_v2(input_image, debug=True):
	'''
	convert a rgb image to a hsv image, using PIL package, not compatible with opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		hsv_image: 		an uint8 hsv numpy image
	'''
	np_image = safe_image(input_image)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use PIL'

	pil_rgb_img = Image.fromarray(np_image)
	pil_hsv_img = pil_rgb_img.convert('HSV')
	hsv_img = np.array(pil_hsv_img)
	return hsv_img

def hsv2rgb(input_image, debug=True):
	'''
	convert a hsv image to a rgb image using opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		rgb_img: 		an uint8 rgb numpy image
	'''
	np_image = safe_image(input_image)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	rgb_img = cv2.cvtColor(np_image, cv2.COLOR_HSV2RGB)
	return rgb_img

def rgb2lab(input_image, debug=True):
	'''
	convert a rgb image to a lab image using opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		lab_image: 		an uint8 lab numpy image
	'''
	np_image = safe_image(input_image)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	lab_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
	return lab_img

def lab2rgb(input_image, debug=True):
	'''
	convert a lab image to a rgb image using opencv package

	parameters:
		input_image:	an pil or numpy image

	output:
		rgb_img: 		an uint8 rgb numpy image
	'''
	np_image = safe_image(input_image)
	if isfloatimage(np_image):
		np_image = (np_image * 255.).astype('uint8')	

	if debug:
		assert iscolorimage(np_image), 'the input image should be a rgb image'
		assert isuintimage(np_image), 'the input numpy image should be uint8 image in order to use opencv'

	rgb_img = cv2.cvtColor(np_image, cv2.COLOR_LAB2RGB)
	return rgb_img

def image_hist_equalization_hsv(input_image, debug=True):
	'''
	do histogram equalization for an image: could be a color image or gray image

	parameters:
		input_image:		an pil or numpy image

	output:
		equalized_image:	an uint8 numpy image (rgb or gray)
	'''
	np_image = safe_image(input_image)
	if isuintimage(np_image):
		np_image = np_image.astype('float32') / 255.

	if debug:
		assert isfloatimage(np_image), 'the input image should be a float image'

	if iscolorimage(np_image):
		hsv_image = rgb2hsv(np_image, debug=debug)
		input_data = hsv_image[:, :, 2]			# extract the value channel
		equalized_hsv_image = (hist_equalization(input_data, num_bins=256, debug=debug) * 255.).astype('uint8')
		hsv_image[:, :, 2] = equalized_hsv_image
		equalized_image = hsv2rgb(hsv_image, debug=debug)
	elif isgrayimage(np_image):
		equalized_image = (hist_equalization(np_image, num_bins=256, debug=debug) * 255.).astype('uint8')
	else:
		assert False, 'the input image is neither a color image or a grayscale image'

	return equalized_image

def image_hist_equalization_lab(input_image, debug=True):
	'''
	do histogram equalization for an image: could be a color image or gray image

	parameters:
		input_image:		an pil or numpy image

	output:
		equalized_image:	an uint8 numpy image (rgb or gray)
	'''
	np_image = safe_image(input_image)
	if isuintimage(np_image):
		np_image = np_image.astype('float32') / 255.

	if debug:
		assert isfloatimage(np_image), 'the input image should be a float image'

	if iscolorimage(np_image):
		hsv_image = rgb2lab(np_image, debug=debug)
		input_data = hsv_image[:, :, 0]			# extract the value channel
		equalized_hsv_image = (hist_equalization(input_data, num_bins=256, debug=debug) * 255.).astype('uint8')
		hsv_image[:, :, 0] = equalized_hsv_image
		equalized_image = lab2rgb(hsv_image, debug=debug)
	elif isgrayimage(np_image):
		equalized_image = (hist_equalization(np_image, num_bins=256, debug=debug) * 255.).astype('uint8')
	else:
		assert False, 'the input image is neither a color image or a grayscale image'

	return equalized_image

# # to test, supposed to be equivalent to gray2rgb
# def mat2im(mat, cmap, limits):
#   '''
# % PURPOSE
# % Uses vectorized code to convert matrix "mat" to an m-by-n-by-3
# % image matrix which can be handled by the Mathworks image-processing
# % functions. The the image is created using a specified color-map
# % and, optionally, a specified maximum value. Note that it discards
# % negative values!
# %
# % INPUTS
# % mat     - an m-by-n matrix  
# % cmap    - an m-by-3 color-map matrix. e.g. hot(100). If the colormap has 
# %           few rows (e.g. less than 20 or so) then the image will appear 
# %           contour-like.
# % limits  - by default the image is normalised to it's max and min values
# %           so as to use the full dynamic range of the
# %           colormap. Alternatively, it may be normalised to between
# %           limits(1) and limits(2). Nan values in limits are ignored. So
# %           to clip the max alone you would do, for example, [nan, 2]
# %          
# %
# % OUTPUTS
# % im - an m-by-n-by-3 image matrix  
#   '''
#   assert len(mat.shape) == 2
#   if len(limits) == 2:
#     minVal = limits[0]
#     tempss = np.zeros(mat.shape) + minVal
#     mat    = np.maximum(tempss, mat)
#     maxVal = limits[1]
#     tempss = np.zeros(mat.shape) + maxVal
#     mat    = np.minimum(tempss, mat)
#   else:
#     minVal = mat.min()
#     maxVal = mat.max()
#   L = len(cmap)
#   if maxVal <= minVal:
#     mat = mat-minVal
#   else:
#     mat = (mat-minVal) / (maxVal-minVal) * (L-1)
#   mat = mat.astype(np.int32)
  
#   image = np.reshape(cmap[ np.reshape(mat, (mat.size)), : ], mat.shape + (3,))
#   return image

# def jet(m):
#   cm_subsection = linspace(0, 1, m)
#   colors = [ cm.jet(x) for x in cm_subsection ]
#   J = np.array(colors)
#   J = J[:, :3]
#   return J

# def generate_color_from_heatmap(maps, num_of_color=100, index=None):
#   assert isinstance(maps, np.ndarray)
#   if len(maps.shape) == 3:
#     return generate_color_from_heatmaps(maps, num_of_color, index)
#   elif len(maps.shape) == 2:
#     return mat2im( maps, jet(num_of_color), [maps.min(), maps.max()] )
#   else:
#     assert False, 'generate_color_from_heatmap wrong shape : {}'.format(maps.shape)

############################################# format transform #################################
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

# done
def	unnormalize_npimage(input_image, img_type='uint8', debug=True):
	'''
	un-normalize an image to an uint8 with range of [0, 255]
	note that: the input might not be an image because the value range might be arbitrary
	'''
	if ispilimage(input_image):
		np_image = np.array(input_image)
	elif isnparray(input_image):
		np_image = input_image.copy()
	else:
		assert False, 'only pil image and numpy array are supported'

	if debug:
		assert isnpimage_dimension(np_image), 'the input numpy image is not correct: {}'.format(np_image.shape)
		assert img_type in {'uint8'}, 'the image does not contain a good type'

	min_val = float(np.min(np_image))
	max_val = float(np.max(np_image))
	if math.isnan(min_val) or math.isnan(max_val):			# with nan
		assert False, 'the input image has nan'
	elif min_val == max_val:								# all same
		np_image.fill(0)
	else:													# normal case
		np_image = np_image - min_val
		np_image = np_image / (max_val - min_val)
		np_image = np_image * 255.

		if debug:
			assert np.min(np_image) == 0 and np.max(np_image) == 255, 'the value range is not right [%f, %f]' % (np.min(np_image), np.max(np_image))

	unnormalized_img = np_image.astype(img_type)
	return unnormalized_img

############################################# 2D transformation #################################
def rotate_bound(image, angle):
    # angle is counter_clockwise
    if angle == -90:
        # rotate clockwise
        return np.rot90(image, 3)
    else:
        return np.rot90(image)
        # rotate counter-clockwise

def pad_around(input_image, pad_rect, pad_value=0, debug=True):
	'''
	this function is to pad given value to an image in provided region, all images in this function are floating images
	
	parameters:
		input_image:	an pil or numpy image
	  	pad_rect:   	a list of 4 non-negative integers, describing how many pixels to pad. The order is [left, top, right, bottom]
	  	pad_value:  	an intger between [0, 255]

	outputs:
		img_padded:		an uint8 numpy image with padding
	'''
	np_image = safe_image(input_image)
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

def crop_center(input_image, center_rect, pad_value=0, debug=True):
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
	np_image = safe_image(input_image)
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
		img_cropped = pad_around(img_cropped, pad_rect=pad_rect, pad_value=pad_value, debug=debug)
	if len(img_cropped.shape) == 3 and img_cropped.shape[2] == 1: img_cropped = img_cropped[:, :, 0]

	return img_cropped, crop_bbox, crop_bbox_clipped

def imresize(img, portion, interp='bicubic', debug=True):
	if debug:
		assert interp == 'bicubic' or interp == 'bilinear', 'the interpolation method is not correct'
		assert isnparray(img), 'the input image is not correct'

	height, width = img.shape[:2]
	if interp == 'bicubic':
	    img_ = cv2.resize(img, (int(portion*width), int(portion*height)), interpolation = cv2.INTER_CUBIC)
	elif interp == 'bilinear':
		img_ = cv2.resize(img, (int(portion*width), int(portion*height)), interpolation = cv2.INTER_LINEAR)
	else:
		assert False, 'interpolation is wrong'

	return img_

############################################# batch processing #################################
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

# done
def draw_mask(np_image, np_image_mask, alpha=0.3, debug=True):
	'''
	draw a mask on top of an image with certain transparency
	'''
	if debug:
		assert isnpimage_dimension(np_image), 'the input image is not correct: {}'.format(np_image.shape)
		assert isnpimage_dimension(np_image_mask), 'the input mask image is not correct: {}'.format(np_image_mask.shape)

	pil_image = Image.fromarray(np_image.copy())
	pil_image_mask = Image.fromarray(np_image_mask.copy())
	pil_image_out = Image.blend(pil_image, pil_image_mask, alpha=alpha)

	return pil_image_out