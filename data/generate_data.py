# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains function for generating many formats of data

from cv2 import imread
import numpy as np
import os, time
import h5py

import __init__paths__
from math_function import identity
from check import is_path_exists, isnparray, is_path_exists_or_creatable, isfile, isfolder, isfunction, isdict
from file_io import file_abspath, load_list_from_file, mkdir_if_missing, fileparts

def generate_hdf5(save_dir, data_src, batch_size=1, ext_filter='png', label_src=None, label_preprocess_function=identity, debug=False):
    '''
    # this function creates data in hdf5 format from a image path 

    # input parameter
    #   data_src:       where the data is
    #   label_src:      where the label is
    #   save_dir:       where to store the hdf5 data
    #   batch_size:     how many image to store in a single hdf file
    #   ext_filder:     what format of data to use for generating hdf5 data 
    '''

    # parse input
    assert is_path_exists_or_creatable(save_dir), 'save path should be a folder to save all hdf5 files'
    mkdir_if_missing(save_dir)

    if isfolder(data_src):
        if debug:
            print 'data is loading from %s' % data_src
        filepath = file_abspath()
        datalist_name = os.path.abspath('./datalist.txt')
        cmd = 'th %s/../file_io/generate_list.lua %s %s %s' % (filepath, data_src, datalist_name, ext_filter)
        print cmd
        os.system(cmd)    # generate data list
        datalist, num_data = load_list_from_file(datalist_name)
        os.system('rm %s' % datalist_name)
    elif isfile(data_src):
        datalist, num_data = load_list_from_file(data_src)
    else:
        assert(False, 'data source format is not correct.')
    
    if label_src is None:
        labeldict = None
    elif isfile(label_src):
        assert is_path_exists(label_src), 'file not found'
        _, _, ext = fileparts(label_src)
        assert ext == '.json', 'only json extension is supported'
        labeldict = json.load(label_src)
        num_label = len(labeldict)
        assert(num_data == num_label, 'number of data and label is not equal.')
    elif isdict(label_src):
        labeldict = label_src;
    else:
        assert False, 'label source format is not correct.'
    
    assert isfunction(label_preprocess_function), 'label preprocess function is not correct.'

    # warm up
    size_data = imread(datalist[0]).shape
    data = np.zeros(size_data + (batch_size, ), dtype='float32')
    if labeldict is not None:
        labels = np.zeros([1, batch_size], dtype='float32')
        label_value = [float(label_tmp_char) for label_tmp_char in labeldict.values()]
        label_range = np.array([min(label_value), max(label_value)])

    # start generating
    count_hdf = 1       # count number of hdf5 file
    for i in xrange(num_data):
        print('%s %d/%d' % (save_dir, i+1, num_data))
        img = imread(datalist[i]).astype('float32') / 255.0   # [rows,col,channel,numbers], scale the image data to (0, 1)
        if batch_size > 1:
            assert size_data == img.shape, 'image size should be equal in each single hdf5 file.'
        
        size_data = img.shape
        data[:, :, :, i % batch_size] = img

        if labeldict is not None:
            _, name, _ = fileparts(datalist[i])
            labels[0, i % batch_size] = float(labeldict[name])

        if i % batch_size == 0:
            # preprocess
            data = data[:, :, [2, 1, 0], :]        # from rgb to brg
            data = np.transpose(data, (1, 0, 2, 3))        # permute to [cols, rows, channel, numbers]

            # write to hdf5 format
            h5f = h5py.File('%s/data_%010d.hdf5' % (save_dir, count_hdf), 'w')
            h5f.create_dataset('data', data=data, dtype='float32')
            if labeldict is not None:
                labels = label_preprocess_function(data=labels, data_range=label_range)
                h5f.create_dataset('label', data=labels, dtype='float32')
                labels = np.zeros([1, batch_size], dtype='float32')

            h5f.close()
            count_hdf = count_hdf + 1
            data = np.zeros(size_data + (batch_size, ), dtype='float32')

    return num_hdf5, num_data