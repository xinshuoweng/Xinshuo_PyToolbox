# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import cv2, os
from skvideo.io import FFmpegWriter

from xinshuo_miscellaneous import is_path_exists, islistofstring, ispositiveinteger, reverse_list
from xinshuo_visualization import visualize_image
from xinshuo_io import mkdir_if_missing, load_image, load_list_from_folder
from xinshuo_images import image_resize

def extract_images_from_video_opencv(video_file, save_dir, debug=True):
	'''
	if the VideoCapture does not work, uninstall python-opencv and reinstall the newest version
	'''
	if debug: assert is_path_exists(video_file), 'the input video file does not exist'
	mkdir_if_missing(save_dir)
	cap = cv2.VideoCapture(video_file)
	frame_id = 0

	while(True):
		ret, frame = cap.read()
		if not ret: break
		save_path = os.path.join(save_dir, 'image%05d.png' % frame_id)
		visualize_image(frame, bgr2rgb=True, save_path=save_path)
		frame_id += 1
		print('processing frame %d' % frame_id)

	cap.release()

def extract_images_from_video_ffmpeg(video_file, save_dir, format='frame%06d.png', debug=True):
	'''
	loading the video using the ffmpeg
	'''
	if debug: assert is_path_exists(video_file), 'the input video file does not exist'
	mkdir_if_missing(save_dir)
	command = 'ffmpeg -i %s %s/%s' % (video_file, save_dir, format)
	os.system(command)

def extract_images_from_video_ffmpeg2(video_file, save_dir, format='frame%06d.png', debug=True):
    """loading the video using the ffmpeg built-in in python.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""

    vid = imageio.get_reader(filename, 'ffmpeg')
    frames = []
    for i in range(0, num_frames):
        image = vid.get_data(i)
        image = functional.to_tensor(image)
        frames.append(image)
    return frames

def generate_video_from_list(image_list, save_path, framerate=30, downsample=1, warning=True, debug=True):
	'''
	create video from a list of images with a framerate
	note that: the height and widht of the images should be a multiple of 2

	parameters:
		image_list:			a list of image path
		save_path:			the path to save the video file
		framerate:			fps 
	'''
	if debug: 
		assert islistofstring(image_list), 'the input is not correct'
		assert ispositiveinteger(framerate), 'the framerate is a positive integer'
	mkdir_if_missing(save_path)
	inputdict = {'-r': str(framerate)}
	outputdict = {'-r': str(framerate), '-crf': '18', '-vcodec': 'libx264', '-profile:V': 'high', '-pix_fmt': 'yuv420p'}
	video_writer = FFmpegWriter(save_path, inputdict=inputdict, outputdict=outputdict)
	count = 1
	num_images = len(image_list)
	for image_path in image_list:
		print('processing frame %d/%d' % (count, num_images))
		image = load_image(image_path, resize_factor=downsample, warning=warning, debug=debug)

		# make sure the height and width are multiple of 2
		height, width = image.shape[0], image.shape[1]
		if not (height % 2 == 0 and width % 2 == 0):
			height += height % 2
			width += width % 2
			image = image_resize(image, target_size=[height, width], warning=warning, debug=debug)

		video_writer.writeFrame(image)
		count += 1

	video_writer.close()

def generate_video_from_folder(images_dir, save_path, framerate=30, downsample=1, reverse=False, warning=True, debug=True):
	image_list, num_images = load_list_from_folder(images_dir, ext_filter=['.jpg', '.png', '.jpeg'], debug=debug)
	if reverse: image_list = reverse_list(image_list, warning=warning, debug=debug)
	print('%d images loaded' % num_images)
	generate_video_from_list(image_list, save_path, framerate=framerate, downsample=downsample, warning=warning, debug=debug)