# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import cv2, os

from xinshuo_python import is_path_exists
from xinshuo_io import mkdir_if_missing
from xinshuo_visualization import visualize_image

# done
def convert_video2images(video_file, save_dir, debug=True):
	'''
	if the VideoCapture does not work, uninstall python-opencv and reinstall the newest version
	'''
	if debug:
		assert is_path_exists(video_file), 'the input video file does not exist'
	
	mkdir_if_missing(save_dir)
	cap = cv2.VideoCapture(video_file)
	frame_id = 0

	while(True):
		ret, frame = cap.read()
		if not ret:
			break
		save_path = os.path.join(save_dir, 'image%05d.png' % frame_id)
		visualize_image(frame, is_cvimage=True, save_path=save_path)
		frame_id += 1
		print('processing frame %d' % frame_id)

	cap.release()