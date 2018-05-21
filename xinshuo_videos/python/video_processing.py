# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import cv2, os
from skvideo.io import FFmpegWriter

from xinshuo_miscellaneous import is_path_exists, islistofstring, ispositiveinteger
from xinshuo_visualization import visualize_image
from xinshuo_io import mkdir_if_missing, load_image

def extract_images_from_video(video_file, save_dir, debug=True):
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

def generate_video_from_list(image_list, save_path, framerate=30, warning=True, debug=True):
	if debug: 
		assert islistofstring(image_list), 'the input is not correct'
		assert ispositiveinteger(framerate), 'the framerate is a positive integer'
	mkdir_if_missing(save_path)
	inputdict = {'-r': str(framerate)}
	outputdict = {'-r': str(framerate), '-crf': '18', '-c:v': 'libx264', '-profile:V': 'high', '-pix_fmt': 'yuv420p'}

	video_writer = FFmpegWriter(save_path, inputdict=inputdict, outputdict=outputdict)
	for image_path in image_list:
		image = load_image(image_path, warning=warning, debug=debug)
		video_writer.writeFrame(image)