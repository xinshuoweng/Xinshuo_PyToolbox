# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import math, cv2
import numpy as np
from PIL import Image

from xinshuo_python import *
from xinshuo_vision import clip_bboxes_TLWH

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

############################################# image conversion #################################

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
def	unnormalize_npimage(np_image, img_type='uint8', debug=True):
	'''
	un-normalize a numpy image and scale it
		for uint8, scaled to [0, 255]
	'''
	if debug:
		assert isnpimage_dimension(np_image), 'the input numpy image is not correct: {}'.format(np_image.shape)
		assert img_type in {'uint8'}, 'the image does not contain a good type'

	unnormalized_img = np_image.copy()
	min_val = np.min(unnormalized_img)
	max_val = np.max(unnormalized_img)
	if math.isnan(min_val) or math.isnan(max_val):			# with nan
		assert False, 'the input image has nan'
	elif min_val == max_val:								# all same
		unnormalized_img.fill(0)
	else:													# normal case
		unnormalized_img = unnormalized_img - min_val
		unnormalized_img = unnormalized_img / (max_val - min_val)
		unnormalized_img = unnormalized_img * 255.
		if debug:
			assert np.min(unnormalized_img) == 0 and np.max(unnormalized_img) == 255, 'the value range is not right [%f, %f]' % (min_val, max_val)

	unnormalized_img = unnormalized_img.astype(img_type)
	return unnormalized_img

# done
def gray2rgb(np_image, with_color=True, cmap='jet', debug=True):
	'''
	convert a grayscale image (1-channel) to a rgb image
		with_color:	add colormap
	'''
	if debug:
		assert isgrayimage(np_image), 'the input numpy image is not correct: {}'.format(np_image.shape)

	if with_color:
		if cmap == 'jet':
			rgb_image = cv2.applyColorMap(np_image, cv2.COLORMAP_JET)
		else:
			assert False, 'cmap %s is not supported' % cmap
	else:
		rgb_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)

	return rgb_image

############################################## miscellaneous

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


# TODO
def crop_center(img1, rect, pad_value):
	# rect is XYWH, only uint8 is supported

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
    cropped = cropped.astype('uint8')

    # TODO: with padding
    [im_height, im_width, im_channel] = img1.shape
    
    crop_rect_ori = clip_bboxes_TLWH(crop_rect, im_width, im_height)
    
    crop_rect = np.array(crop_rect).reshape((1, 4))
    crop_rect_ori = np.array(crop_rect_ori).reshape((1, 4))
    return cropped, crop_rect, crop_rect_ori


def imresize(img, portion, interp='bicubic', debug=True):
	if debug:
		assert interp == 'bicubic' or interp == 'bilinear', 'the interpolation method is not correct'

	height, width = img.shape[:2]
	if interp == 'bicubic':
	    img_ = cv2.resize(img, (int(portion*width), int(portion*height)), interpolation = cv2.INTER_CUBIC)
	elif interp == 'bilinear':
		img_ = cv2.resize(img, (int(portion*width), int(portion*height)), interpolation = cv2.INTER_LINEAR)
	else:
		assert False, 'interpolation is wrong'

	return img_


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

def rotate_bound(image, angle):
    # angle is counter_clockwise
    if angle == -90:
        # rotate clockwise
        return np.rot90(image, 3)
    else:
        return np.rot90(image)
        # rotate counter-clockwise


# done
def draw_mask(np_image, np_image_mask, alpha=0.3, debug=True):
	'''
	draw a mask on top of an image with certain transparency
	'''
	if debug:
		assert isnpimage_dimension(np_image), 'the input image is not correct: {}'.format(np_image.shape)
		assert isnpimage_dimension(np_image_mask), 'the input mask image is not correct: {}'.format(np_image_mask.shape)

	pil_image = Image.fromarray(np_image)
	pil_image_mask = Image.fromarray(np_image_mask)
	pil_image_out = Image.blend(pil_image, pil_image_mask, alpha=alpha)

	return pil_image_out