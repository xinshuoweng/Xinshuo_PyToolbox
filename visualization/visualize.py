# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn.neighbors import NearestNeighbors
from scipy.misc import imread
import sys

import __init__paths__
from check import *
from file_io import mkdir_if_missing, fileparts
from bbox_transform import bbox_TLBR2TLWH, bboxcheck_TLBR

color_set = ['b', 'g', 'y', 'k', 'r']

def visualize_save_image(image, vis=True, save=False, save_path=None, debug=True):
    '''
    input image is a numpy array matrix
    '''
    if islist(image):
        imagelist = image
        save_path_list = save_path
        if vis:
            print('visualizing a list of images:')
        if save:
            print('saving a list of images')
            if debug:
                assert islist(save_path_list), 'for saving a list of images, please provide a list of saving path'
                assert all(is_path_exists_or_creatable(save_path_tmp) and isfile(save_path_tmp) for save_path_tmp in save_path_list), 'save path is not valid'
                assert len(save_path_list) == len(imagelist), 'length of list for saving path and data is not equal'
        index = 0
        for image_tmp in imagelist:
            print('processing %d/%d' % (index+1, len(imagelist)))
            if save:
                visualize_save_image(image_tmp, vis=vis, save_path=save_path[i], save=save, debug=debug)
            else:
                visualize_save_image(image_tmp, vis=vis, debug=debug)
            index += 1
        return

    if debug:
        assert isnparray(image), 'input image is not a numpy array {}'.format(type(image))
        assert isimage(image, debug=debug), 'input is not a good image, shape is {}'.format(image.shape)

    dpi = 80  
    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    if iscolorimage(image, debug=debug):
        if debug:
            print 'visualizing color image'
        ax.imshow(image, interpolation='nearest')
    elif isgrayimage(image, debug=debug):
        if debug:
            print 'visualizing grayscale image'
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.reshape(image, image.shape[:-1])

        if isfloatimage(image, debug=debug) and all(item == 1.0 for item in image.flatten().tolist()):
            if debug:
                print('all elements in image are 1. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 0.00001
        elif isuintimage(image, debug=debug) and all(item == 255 for item in image.flatten().tolist()):
            if debug:
                print('all elements in image are 255. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 1
        ax.imshow(image, interpolation='nearest', cmap='gray')
    else:
        assert False, 'image is not correct'
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    if save:
        if debug:
            assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not valid: %s' % save_path
            mkdir_if_missing(save_path)
        fig.savefig(save_path, dpi=dpi, transparent=True)
    if vis:
        plt.show()

    plt.close(fig)
    return

def visualize_image_with_pts(image_path, pts, vis=True, save=False, save_path=None, debug=True):
    '''
    visualize image and plot keypoints on top of it

    parameter:
        image_path:     a path to an image
        pts:            2 x num_pts numpy array
    '''

    if debug:
        assert is_path_exists(image_path), 'image is not existing'
        assert isnparray(pts) and pts.shape[0] == 2, 'input points are not correct'

    try:
        image = imread(image_path)
    except IOError:
        print('path is not a valid image path. Please check: %s' % image_path)
        return

    dpi = 80  
    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # display image
    if iscolorimage(image, debug=debug):
        if debug:
            print 'visualizing color image'
        ax.imshow(image, interpolation='nearest')
    elif isgrayimage(image, debug=debug):
        if debug:
            print 'visualizing grayscale image'
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.reshape(image, image.shape[:-1])

        if isfloatimage(image, debug=debug) and all(item == 1.0 for item in image.flatten().tolist()):
            if debug:
                print('all elements in image are 1. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 0.00001
        elif isuintimage(image, debug=debug) and all(item == 255 for item in image.flatten().tolist()):
            if debug:
                print('all elements in image are 255. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 1
        ax.imshow(image, interpolation='nearest', cmap='gray')
    else:
        assert False, 'image is not correct'
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    # plot keypoints
    ax.scatter(pts[0, :], pts[1, :], color='r')

    # save and visualization
    if save:
        if debug:
            assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not valid: %s' % save_path
            mkdir_if_missing(save_path)
        fig.savefig(save_path, dpi=dpi, transparent=True)
    if vis:
        plt.show()
    plt.close(fig)
    return

def visualize_image_with_bbox(image_path, bbox, vis=True, save=False, save_path=None, debug=True):
    '''
    visualize image and plot keypoints on top of it

    parameter:
        image_path:     a path to an image
        bbox:           N X 4 numpy array, with TLBR format
    '''

    if debug:
        assert is_path_exists(image_path), 'image is not existing'
        assert bboxcheck_TLBR(bbox), 'input bounding boxes are not correct'

    try:
        image = imread(image_path)
    except IOError:
        print('path is not a valid image path. Please check: %s' % image_path)
        return

    dpi = 80  
    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # display image
    if iscolorimage(image, debug=debug):
        if debug:
            print 'visualizing color image'
        ax.imshow(image, interpolation='nearest')
    elif isgrayimage(image, debug=debug):
        if debug:
            print 'visualizing grayscale image'
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.reshape(image, image.shape[:-1])
        if isfloatimage(image, debug=debug) and all(item == 1.0 for item in image.flatten().tolist()):
            if debug:
                print('all elements in image are 1. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 0.00001
        elif isuintimage(image, debug=debug) and all(item == 255 for item in image.flatten().tolist()):
            if debug:
                print('all elements in image are 255. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 1
        ax.imshow(image, interpolation='nearest', cmap='gray')
    else:
        assert False, 'image is not correct'
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    # plot bounding box
    bbox = bbox_TLBR2TLWH(bbox)              # convert TLBR format to TLWH format
    for bbox_index in range(bbox.shape[0]):
        bbox_tmp = bbox[bbox_index, :]     
        ax.add_patch(plt.Rectangle((bbox_tmp[0], bbox_tmp[1]), bbox_tmp[2], bbox_tmp[3], fill=False, edgecolor='red', linewidth=3.5))

    # save and visualization
    if save:
        if debug:
            assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not valid: %s' % save_path
            mkdir_if_missing(save_path)
        fig.savefig(save_path, dpi=dpi, transparent=True)
    if vis:
        plt.show()
    plt.close(fig)
    return

def visualize_ced(normed_mean_error_dict, error_threshold, debug=True, vis=True, save=False, save_path=None):
    '''
    visualize the cumulative error distribution curve (alse called NME curve or pck curve)
    all parameters are represented by percentage

    parameter:
        normed_mean_error_dict:     a dictionary whose keys are the method name and values are (N, ) numpy array to represent error in evaluation
        error_threshold:            threshold to display in x axis

    return:
        AUC:                        area under the curve
        MSE:                        mean square error
    '''
    if debug:
        assert isdict(normed_mean_error_dict), 'the input normalized mean error dictionary is not correct'
        assert error_threshold > 0 and error_threshold < 100, 'threshold percentage is not well set'
        if save:
            assert save_path is not None and is_path_exists_or_creatable(save_path), 'please provide a valid path to save the results' 
    
    # set display parameters
    dpi = 80  
    width = 1000
    height = 800
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)

    # set figure handle
    num_bins = 10000
    x_axis = np.linspace(0, 1, num_bins)            # error axis, percentage of normalization factor
    y_axis = np.zeros(num_bins)
    scale = num_bins / 100
    interval_y = 10
    interval_x = 1
    plt.xlim(0, error_threshold)
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.xticks(np.arange(0, error_threshold + interval_x, interval_x))
    plt.grid()
    plt.title('2D PCK curve', fontsize=20)
    plt.xlabel('Normalized distance (%)', fontsize=16)
    plt.ylabel('Test Images (%)', fontsize=16)

    # calculate metrics for each method
    num_methods = len(normed_mean_error_dict)
    num_images = len(normed_mean_error_dict.values()[0])
    AUC = dict()
    MSE = dict()
    method_index = 0
    assert num_images > 0, 'number of error array should be larger than 0'
    for method_name, normed_mean_error in normed_mean_error_dict.items():
        if debug:
            assert isnparray(normed_mean_error) and len(normed_mean_error) == num_images and normed_mean_error.ndim == 1, 'the input error array is not correct'

        for i in range(num_bins):
            y_axis[i] = float((normed_mean_error < x_axis[i]).sum()) / num_images         # percentage of error

        AUC[method_name] = np.sum(y_axis[:error_threshold * scale]) / (error_threshold * scale)              # bigger, better
        MSE[method_name] = np.mean(normed_mean_error)                                                  # smaller, better

        # draw 
        color_index = method_index % len(color_set) 
        color_tmp = color_set[color_index]
        plt.plot(x_axis*100, y_axis*100, '%s-' % color_tmp, label=method_name, lw=3)
        plt.legend(loc=4, fontsize=16)
        method_index += 1

    # print('AUC: %f' % AUC)
    # print('MSE: %f' % MSE)
    if vis:
        plt.show()
    if save:
        plt.savefig(save_path, dpi=dpi, transparent=True)
    plt.close()

    return AUC, MSE


def nearest_neighbor_visualization(featuremap_dict, num_neighbor=5, top_number=5, vis=True, save_csv=False, csv_save_path=None, save_vis=False, save_img=False, save_thumb_name='nearest_neighbor.png', img_src_folder=None, ext_filter='.jpg', nn_save_folder=None, debug=True):
    '''
    visualize nearest neighbor for featuremap from images

    parameter:
        featuremap_dict: a dictionary contains image path as key, and featuremap as value, the featuremap needs to be numpy array with any shape. No flatten needed
        num_neighbor: number of neighbor to visualize, the first nearest is itself
        top_number: number of top to visualize, since there might be tons of featuremap (length of dictionary), we choose the top ten with lowest distance with their nearest neighbor
        csv_save_path: path to save .csv file which contains indices and distance array for all elements
        nn_save_folder: save the nearest neighbor images for top featuremap

    return:
        all_sorted_nearest_id: a 2d matrix, each row is a feature followed by its nearest neighbor in whole feature dataset, the column is sorted by the distance of all nearest neighbor each row
        selected_nearest_id: only top number of sorted nearest id 
    '''
    print('processing feature map to nearest neightbor.......')
    if debug:
        assert isdict(featuremap_dict), 'featuremap should be dictionary'
        assert all(isnparray(featuremap_tmp) for featuremap_tmp in featuremap_dict.values()), 'value of dictionary should be numpy array'
        assert isinteger(num_neighbor) and num_neighbor > 1, 'number of neighborhodd is an integer larger than 1'
        if save_csv and csv_save_path is not None:
            assert is_path_exists_or_creatable(csv_save_path), 'path to save .csv file is not correct'
        
        if save_vis or save_img:
            if nn_save_folder is not None:  # save image directly
                assert isstring(ext_filter), 'extension filter is not correct'
                assert is_path_exists(img_src_folder), 'source folder for image is not correct'
                assert all(isstring(path_tmp) for path_tmp in featuremap_dict.keys())     # key should be the path for the image
                assert is_path_exists_or_creatable(nn_save_folder), 'folder to save top visualized images is not correct'
                assert isstring(save_thumb_name), 'name of thumbnail is not correct'

    if ext_filter.find('.') == -1:
        ext_filter = '.%s' % ext_filter

    # flatten the feature map
    nn_feature_dict = dict()
    for key, featuremap_tmp in featuremap_dict.items():
        nn_feature_dict[key] = featuremap_tmp.flatten()
    num_features = len(nn_feature_dict)

    # nearest neighbor
    featuremap = np.array(nn_feature_dict.values())
    nearbrs = NearestNeighbors(n_neighbors=num_neighbor, algorithm='ball_tree').fit(featuremap)
    distances, indices = nearbrs.kneighbors(featuremap)
    
    if debug:
        assert featuremap.shape[0] == num_features, 'shape of feature map is not correct'
        assert indices.shape == (num_features, num_neighbor), 'shape of indices is not correct'
        assert distances.shape == (num_features, num_neighbor), 'shape of indices is not correct'

    # convert the nearest indices for all featuremap to the key accordingly
    id_list = nn_feature_dict.keys()
    max_length = len(max(id_list, key=len))     # find the maximum length of string in the key
    nearest_id = np.chararray(indices.shape, itemsize=max_length+1)
    for x in range(nearest_id.shape[0]):
        for y in range(nearest_id.shape[1]):
            nearest_id[x, y] = id_list[indices[x, y]]

    if debug:
        assert list(nearest_id[:, 0]) == id_list, 'nearest neighbor has problem'
    
    # sort the feature based on distance
    print('sorting the feature based on distance')
    featuremap_distance = np.sum(distances, axis=1)
    if debug:
        assert featuremap_distance.shape == (num_features, ), 'distance is not correct'
    sorted_indices = np.argsort(featuremap_distance)
    all_sorted_nearest_id = nearest_id[sorted_indices, :]

    # save to the csv file
    if save_csv and csv_save_path is not None:
        print 'Saving nearest neighbor result as .csv to path: %s' % csv_save_path
        with open(csv_save_path, 'w+') as file:
            np.savetxt(file, distances, delimiter=',', fmt='%f')
            np.savetxt(file, all_sorted_nearest_id, delimiter=',', fmt='%s')
            file.close()

    # choose the best to visualize
    selected_sorted_indices = sorted_indices[0:top_number]
    if debug:
        for i in range(num_features-1):
            assert featuremap_distance[sorted_indices[i]] < featuremap_distance[sorted_indices[i+1]], 'feature map is not well sorted based on distance' 
    selected_nearest_id = nearest_id[selected_sorted_indices, :]

    if save_vis:
        fig, axarray = plt.subplots(top_number, num_neighbor)
        for index in range(top_number):
            for nearest_index in range(num_neighbor):
                img_path = os.path.join(img_src_folder, '%s%s'%(selected_nearest_id[index, nearest_index], ext_filter))
                if debug:
                    print('loading image from %s'%img_path)
                img = imread(img_path)
                if isgrayimage(img, debug=debug):
                    axarray[index, nearest_index].imshow(img, cmap='gray')
                elif iscolorimage(img, debug=debug):
                    axarray[index, nearest_index].imshow(img)
                else:
                    assert False, 'unknown error'
                axarray[index, nearest_index].axis('off')
        save_thumb = os.path.join(nn_save_folder, save_thumb_name)
        fig.savefig(save_thumb)
        if vis:
            plt.show()
        plt.close(fig)

    # save top visualization to the folder
    if save_img and nn_save_folder is not None:
        for top_index in range(top_number):
            file_list = selected_nearest_id[top_index]
            save_subfolder = os.path.join(nn_save_folder, file_list[0])
            mkdir_if_missing(save_subfolder)
            for file_tmp in file_list:
                file_src = os.path.join(img_src_folder, '%s%s'%(file_tmp, ext_filter))
                save_path = os.path.join(save_subfolder, '%s%s'%(file_tmp, ext_filter))
                if debug:
                    print('saving %s to %s' % (file_src, save_path))
                shutil.copyfile(file_src, save_path)

    return all_sorted_nearest_id, selected_nearest_id


def distribution_vis(data, bin_size=None, vis=True, save=False, save_path=None, debug=True):
    '''
    visualize the histgram of a data, which can be a dictionary or list or numpy array or tuple
    '''
    if debug:
        assert istuple(data) or isdict(data) or islist(data) or isnparray(data), 'input data is not correct'

    if isdict(data):
        data = data.values()

    if bin_size is None:
        max_value = np.max(data)
        min_value = np.min(data)
        bin_size = (max_value - min_value) / 10
    else:
        try:
            bin_size = int(bin_size)
        except TypeError:
            print('size of bin should be integer')

    # fixed bin size
    bins = np.arange(min(data) - bin_size, max(data) + bin_size, bin_size) # fixed bin size
    plt.xlim([min(data) - bin_size, max(data) + bin_size])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('distribution of data')
    plt.xlabel('data (bin size = %d)' % int(bin_size))
    plt.ylabel('count')

    if vis:
        plt.show()

    if save:
        if debug:
            assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not correct' 
        plt.savefig(save_path)

    plt.close()