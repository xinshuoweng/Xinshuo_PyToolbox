# -*- coding: utf-8 -*-

# Author: Xinshuo Weng
# Email: xinshuo.weng@gmail.com

from __future__ import print_function
import functools

from .type_check import isnparray
# logging

def print_log(print_str, log, same_line=False, display=True):
	'''
	print a string to a log file

	parameters:
		print_str:          a string to print
		log:                a opened file to save the log
		same_line:          True if we want to print the string without a new next line
		display:            False if we want to disable to print the string onto the terminal
	'''
	if display:
		if same_line: print('{}'.format(print_str), end='')
		else: print('{}'.format(print_str))

	if same_line: log.write('{}'.format(print_str))
	else: log.write('{}\n'.format(print_str))
	log.flush()

def print_np_shape(nparray, debug=True):
	'''
	print a string to represent the shape of a numpy array
	'''
	if debug: assert isnparray(nparray), 'input is not a numpy array and does not have any shape'
	return '(%s)' % (functools.reduce(lambda x, y: str(x) + ', ' + str(y), nparray.shape))

def print_torch_size(torch_size):
	print(torch_size)
	dims = len(torch_size)
	string = '['
	for idim in range(dims):
		string = string + ' {}'.format(torch_size[idim])
	return string + ']'

def log(text, log, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print_log(text, log=log)

def printProgressBar(iteration, total, log, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    str_to_print = '%s |%s| %s%%%s' % (prefix, bar, percent, suffix)
    # print(str_to_print)
    # print(str_to_print)
    # zxc
    print_log(str_to_print, log=log, same_line=True)
    # Print New Line on Complete
    if iteration == total: print_log(' ', log=log)