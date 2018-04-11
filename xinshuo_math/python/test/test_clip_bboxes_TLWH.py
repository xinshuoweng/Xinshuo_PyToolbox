# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from bbox_transform import clip_bboxes_TLWH
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_clip_bboxes_TLWH():
	bbox = [1, 1, 10, 10]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	# print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([1, 1, 4, 4]).reshape((1, 4)))
	
	bbox = [-1, 3, 20, 3]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([0, 3, 5, 2]).reshape((1, 4)))

	bbox = [-10, 30, 5, 5]
	clipped = clip_bboxes_TLWH(bbox, 5, 5)
	print(clipped)
	assert CHECK_EQ_NUMPY(clipped, np.array([0, 3, 5, 2]).reshape((1, 4)))

if __name__ == '__main__':
	test_clip_bboxes_TLWH()