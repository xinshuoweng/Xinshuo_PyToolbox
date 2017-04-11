# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os

def suppress_caffe_terminal_log():
	'''
	prevent caffe log to terminal
	0 - debug
	1 - info (still a LOT of outputs)
	2 - warnings
	3 - errors
	'''

	os.environ['GLOG_minloglevel'] = '2' 