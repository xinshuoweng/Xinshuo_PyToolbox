# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from private import safe_bbox
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_safe_bbox():
	bbox = [1, 2, 3, 4]
	good_bbox = safe_bbox(bbox)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, np.array(bbox).reshape((1, 4)))

	bbox = np.array([1, 2, 3, 4])
	good_bbox = safe_bbox(bbox)
	print(bbox.shape)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, np.array(bbox).reshape((1, 4)))

	bbox = np.random.rand(10, 4)
	good_bbox = safe_bbox(bbox)
	print(bbox.shape)
	print(good_bbox.shape)
	assert CHECK_EQ_NUMPY(good_bbox, bbox)

	print('\n\nDONE! SUCCESSFULLY!!\n')
if __name__ == '__main__':
	test_safe_bbox()
