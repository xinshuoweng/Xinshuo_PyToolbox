# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains function for generating many formats of data

from cv2 import imread
import numpy as np
import os

import __init__paths__
from math_function import identity
from check import is_path_exists, isstring, isnparray, is_path_exists_or_creatable
from file_io import file_abspath

def generate_hdf5(save_dir, data_src, batch_size=1, ext_filter='png', label_src=None, label_preprocess_function=identity):
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
    # mkdir_is_missing(save_dir);

    if is_path_exists(data_src):
        filepath = file_abspath()
        datalist_name = 'datalist.txt'
        cmd = 'th %s/../file_io/generate_list %s %s %s' % (filepath, data_src, datalist_name, ext_filter)
        os.system(cmd)    # generate data list
        datalist, num_data = load_list_from_file(data_src)
    elif isstring(data_src):
        datalist, num_data = load_list_from_file(data_src)
    else:
        assert(False, 'data source format is not correct.')
    

    if label_src is None:
        labellist = None
    elif isstring(label_src):
        labellist, num_label = load_list_from_file(label_src)
        assert(num_data == num_label, 'number of data and label is not equal.')
        labellist = float(labellist)
        # labellist = cellfun(@(x) str2double(x), labellist, 'UniformOutput', false)
        # labellist = cell2mat(labellist)
    elif isnparray(label_src):
        labellist = label_src;
    else:
        assert False, 'label source format is not correct.'
    
    assert isfunction(label_preprocess_function), 'label preprocess function is not correct.'



    size_data = imread(datalist[1]).shape
    data = np.zeros(size_data + (batch_size, ), dtype='float32')
    if labellist is not None:
        labels = np.zeros([1, batch_size], dtype='float32')

    count_hdf = 1
    for i in xrange(num_data):
        print('%d/%d\n', i, num_data)
        img = imread(datalist[i]).astype('float32')    # [rows,col,channel,numbers], scale the image data to (0, 1)
        if batch_size > 1:
            assert size_data == img.shape, 'image size should be equal in each single hdf5 file.'
        
        size_data = img.shape
        data[:,:,:, mod(i-1, batch_size)+1] = img

        if labellist is not None:
            labels[1, mod(i-1, batch_size)+1] = labellist[i]
        

        if mod(i, batch_size) == 0:
            # preprocess
            data = data[:, :, [3, 2, 1], :]        # from rgb to brg
            data = np.transpose(data, (1, 0, 2, 3))        # permute to [cols, rows, channel, numbers]
            
            # write to hdf5 format
            h5f = h5py.File('%s/data_%10d.hdf5' % (save_dir, count_hdf), 'w')
            h5f.create_dataset('data', data=data, dtype='float32')
    
            # h5write(sprintf('%s/data_%10d.hdf5', save_dir, count_hdf), '/data', data)

            if labellist is not None:
                labels = label_preprocess_function(labels)
                h5f.create_dataset('label', data=labels, dtype='float32')

                # h5create(sprintf('%s/data_%10d.hdf5', save_dir, count_hdf),'/label', size(labels), 'Datatype', 'single')
                # h5write(sprintf('%s/data_%10d.hdf5', save_dir, count_hdf), '/label', labels)
                labels = np.zeros([1, batch_size], dtype='float32')

            # data = zeros([size(img), batch_size]);
            h5f.close()
            count_hdf = count_hdf + 1
            data = np.zeros(size_data + (batch_size, ), dtype='float32')

    return num_hdf5, num_data