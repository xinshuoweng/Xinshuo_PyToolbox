# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains function for generating many formats of data

from scipy.misc import imread
import numpy as np
import os, time
import h5py
import random

import __init__paths__
from math_function import identity
from check import is_path_exists, isnparray, is_path_exists_or_creatable, isfile, isfolder, isfunction, isdict, isstring, islist, isimage
from file_io import load_list_from_file, mkdir_if_missing, fileparts, load_list_from_folder
from preprocess import preprocess_image_caffe
from timer import Timer

def generate_hdf5(save_dir, data_src, data_name='data', batch_size=1, ext_filter='png', label_src=None, label_name='label', label_preprocess_function=identity, debug=True, vis=True):
    '''
    # this function creates data in hdf5 format from a image path 

    # input parameter
    #   data_src:       source of image data, which can be a list of image path, a txt file contains a list of image path, a folder contains a set of images, a list of numpy array image data
    #   label_src:      source of label data, which can be none, a file contains a set of labels, a dictionary of labels, a 1-d numpy array data, a list of label data
    #   save_dir:       where to store the hdf5 data
    #   batch_size:     how many image to store in a single hdf file
    #   ext_filder:     what format of data to use for generating hdf5 data 
    '''

    # parse input
    assert is_path_exists_or_creatable(save_dir), 'save path should be a folder to save all hdf5 files'
    mkdir_if_missing(save_dir)
    assert isstring(data_name), 'dataset name is not correct'   # name for hdf5 data

    # convert data source to a list of numpy array image data
    if isfolder(data_src):
        if debug:
            print 'data is loading from %s with extension .%s' % (data_src, ext_filter)
        filelist, num_data = load_list_from_folder(data_src, ext_filter=ext_filter)
        datalist = None
    elif isfile(data_src):
        filelist, num_data = load_list_from_file(data_src)
        datalist = None
    elif islist(data_src):
        if debug:
            assert all(isimage(data_tmp) for data_tmp in data_src), 'input data source is not a list of numpy array image data'
        datalist = data_src
        num_data = len(datalist)
        filelist = None
    else:
        assert False, 'data source format is not correct.'
    if filelist is not None and datalist is None:
        if debug:
            assert islist(filelist), 'file list is not correct'
        datalist = list()

        # read image from list of path and become a list of image numpy array data
        for imagefile in filelist:
            raw_image = imread(imagefile).astype('float32')
            img = raw_image / 255.0   # [rows,col,channel,numbers], scale the image data to (0, 1)
            if debug:
                max_value = np.max(raw_image)
                assert max_value > 1 and max_value <= 255, 'data is not in [0, 255]'
                min_value = np.min(img)
                max_value = np.max(img)
                assert max_value >= 0 and max_value <= 1, 'data is not in [0, 1]'
                assert min_value >= 0 and min_value <= 1, 'data is not in [0, 1]'
            datalist.append(img)
    if debug:
        assert len(datalist) == num_data, 'number of data is not equal'

    # convert label source to a list of numpy array label
    if label_src is None:
        labeldict = None
        labellist = None
    elif isfile(label_src):
        assert is_path_exists(label_src), 'file not found'
        _, _, ext = fileparts(label_src)
        assert ext == '.json', 'only json extension is supported'
        labeldict = json.load(label_src)
        num_label = len(labeldict)
        assert num_data == num_label, 'number of data and label is not equal.'
        labellist = None
    elif isdict(label_src):
        labeldict = label_src
        labellist = None
    elif isnparray(label_src):
        if debug:
            assert label_src.ndim == 1, 'only 1-d label is supported'
        labeldict = None
        labellist = label_src
    elif islist(label_src):
        if debug:
            assert all(np.array(label_tmp).size == 1 for label_tmp in label_src), 'only 1-d label is supported'
        labellist = label_src
        labeldict = None
    else:
        assert False, 'label source format is not correct.'
    
    assert isfunction(label_preprocess_function), 'label preprocess function is not correct.'

    # warm up
    size_data = datalist[0].shape
    datalist_batch = list()
    if labeldict is not None:
        assert isstring(label_name), 'label name is not correct'
        labels = np.zeros((batch_size, 1), dtype='float32')
        label_value = [float(label_tmp_char) for label_tmp_char in labeldict.values()]
        label_range = np.array([min(label_value), max(label_value)])
    if labellist is not None:
        labels = np.zeros((batch_size, 1), dtype='float32')
        label_range = [np.min(labellist), np.max(labellist)]

    # start generating
    count_hdf = 1       # count number of hdf5 file
    clock = Timer()
    for i in xrange(num_data):
        clock.tic()
        if debug:
            assert size_data == datalist[i].shape

        datalist_batch.append(datalist[i])

        if labeldict is not None and labellist is None:
            if debug:
                assert not filelist, 'file list is empty'
            _, name, _ = fileparts(filelist[i])
            labels[i % batch_size, 0] = float(labeldict[name])
        elif labellist is not None and labeldict is None:
            labels[i % batch_size, 0] = float(labellist[i])
        else:
            assert False, 'label is not correct'

        if i % batch_size == 0:
            data = preprocess_image_caffe(datalist_batch, debug=debug, vis=vis)   # swap channel, transfer from list of HxWxC to NxCxHxW

            # write to hdf5 format
            h5f = h5py.File(os.path.join(save_dir, 'data_%010d.hdf5' % count_hdf), 'w')
            h5f.create_dataset(data_name, data=data, dtype='float32')
            if (labeldict is not None) or (labellist is not None):
                labels = label_preprocess_function(data=labels, data_range=label_range, debug=debug)
                h5f.create_dataset(label_name, data=labels, dtype='float32')
                labels = np.zeros((batch_size, 1), dtype='float32')

            h5f.close()
            count_hdf = count_hdf + 1
            del datalist_batch[:]
            if debug:
                assert len(datalist_batch) == 0, 'list has not been cleared'
        average_time = clock.toc()
        print('saving to %s: %d/%d, average time:%.3f, elapsed time:%.3f, estimated time remaining:%.3f' % (save_dir, i+1, num_data, average_time, average_time * i, average_time * (len(num_data) - i)))

    
    return count_hdf-1, num_data



def load_hdf5_data(hdf5_src, dataname):
    assert is_path_exists(hdf5_src) and isfolder(hdf5_src), 'input hdf5 path does not exist'
    assert islist(dataname), 'dataset queried is not correct'
    assert all(isstring(dataset_tmp) for dataset_tmp in dataname), 'dataset queried is not correct'

    hdf5list, num_hdf5_files = load_list_from_folder(folder_path=hdf5_src, ext_filter='.hdf5')
    check_index = random.randrange(0, num_hdf5_files)
    hdf5_path_sample = hdf5list[check_index]
    hdf5_file = h5py.File(hdf5_path_sample, 'r')
    datadict = dict()
    for dataset in dataname:
        datadict[dataset] = hdf5_file[dataset]
    return datadict