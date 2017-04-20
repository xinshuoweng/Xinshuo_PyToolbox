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

def visualize_save_image(image, vis=True, save=False, save_path=None, debug=True):
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
        assert isimage(image), 'input is not a good image, shape is {}'.format(image.shape)

    dpi = 80  
    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')


    if iscolorimage(image):
        if debug:
            print 'visualizing color image'
        ax.imshow(image, interpolation='nearest')
    elif isgrayimage(image):
        if debug:
            print 'visualizing grayscale image'
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.reshape(image, image.shape[:-1])

        if isfloatimage(image) and all(item == 1.0 for item in image.flatten().tolist()):
            if debug:
                print('all elements in image are 1. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 0.00001
        elif isuintimage(image) and all(item == 255 for item in image.flatten().tolist()):
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



def nearest_neighbor_visualization(featuremap_dict, num_neighbor=5, top_number=5, vis=True, save=False, csv_save_path=None, save_thumb_name='nearest_neighbor.png', img_src_folder=None, ext_filter='.jpg', nn_save_folder=None, debug=True):
    '''
    visualize nearest neighbor for featuremap from images

    parameter:
        featuremap_dict: a dictionary contains image path as key, and featuremap as value, the featuremap needs to be numpy array with any shape. No flatten needed
        num_neighbor: number of neighbor to visualize, the first nearest is itself
        top_number: number of top to visualize, since there might be tons of featuremap (length of dictionary), we choose the top ten with lowest distance with their nearest neighbor
        csv_save_path: path to save .csv file which contains indices and distance array for all elements
        nn_save_folder: save the nearest neighbor images for top featuremap
    '''
    print('processing feature map to nearest neightbor.......')
    if debug:
        assert isdict(featuremap_dict), 'featuremap should be dictionary'
        assert all(isnparray(featuremap_tmp) for featuremap_tmp in featuremap_dict.values()), 'value of dictionary should be numpy array'
        assert isinteger(num_neighbor) and num_neighbor > 1, 'number of neighborhodd is an integer larger than 1'
        if save:
            if csv_save_path is not None:
                assert is_path_exists_or_creatable(csv_save_path), 'path to save .csv file is not correct'
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


    # save to the csv file
    if save and csv_save_path is not None:
        print 'Saving nearest neighbor result as .csv to path: %s' % csv_save_path
        with open(csv_save_path, 'w+') as file:
            np.savetxt(file, distances, delimiter=',', fmt='%f')
            np.savetxt(file, nearest_id[sorted_indices, :], delimiter=',', fmt='%s')
            file.close()

    # choose the best to visualize
    selected_sorted_indices = sorted_indices[0:top_number]
    if debug:
        for i in range(num_features-1):
            assert featuremap_distance[sorted_indices[i]] < featuremap_distance[sorted_indices[i+1]], 'feature map is not well sorted based on distance' 
    selected_nearest_id = nearest_id[selected_sorted_indices, :]

    if save:
        fig, axarray = plt.subplots(top_number, num_neighbor)
        for index in range(top_number):
            for nearest_index in range(num_neighbor):
                img_path = os.path.join(img_src_folder, '%s%s'%(selected_nearest_id[index, nearest_index], ext_filter))
                if debug:
                    print('loading image from %s'%img_path)
                img = imread(img_path)
                if isgrayimage(img):
                    axarray[index, nearest_index].imshow(img, cmap='gray')
                elif iscolorimage(img):
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
    if save and nn_save_folder is not None:
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