# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import time, shutil, colorsys, cv2
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as plycollections
from matplotlib.patches import Ellipse
import numpy as np
import seaborn as sns
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors
from scipy.misc import imread
from scipy.stats import norm, chi2
from terminaltables import AsciiTable
from collections import Counter

# this file define a set of functions related to matplotlib
from xinshuo_io import mkdir_if_missing, fileparts
from xinshuo_vision import bbox_TLBR2TLWH, bboxcheck_TLBR, get_centered_bbox
from xinshuo_miscellaneous import print_np_shape, list2tuple, list_reorder, remove_list_from_list, scalar_list2str_list, istuple, isdict, islistoflist, islist, isnparray, isstring, ispilimage, iscolorimage, isgrayimage, isfloatimage, isuintimage
from xinshuo_math import pts_euclidean, calculate_truncated_mse

dpi = 80
color_set = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w', 'lime', 'cyan', 'aqua']
color_set_big = ['aqua', 'azure', 'red', 'black', 'blue', 'brown', 'cyan', 'darkblue', 'fuchsia', 'gold', 'green', 'grey', 'indigo', 'magenta', 'lime', 'yellow', 'white', 'tomato', 'salmon']
marker_set = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
hatch_set = [None, 'o', '/', '\\', '|', '-', '+', '*', 'x', 'O', '.']
linestyle_set = ['-', '--', '-.', ':', None, ' ', 'solid', 'dashed']

# done
def visualize_image(image_path, is_cvimage=False, vis=False, save_path=None, debug=False, warning=False, closefig=True):
    '''
    visualize various images

    parameters:
        image_path:         a path to an image / an image / a list of images / a list of image paths
    '''
    if debug:
        if isstring(image_path):
            assert is_path_exists(image_path), 'image is not existing at %s' % image_path
        else:
            assert islist(image_path) or isimage(image_path, debug=debug), 'the input is not a list or an good image'

    if isstring(image_path):            # a path to an image
        try:
            image = imread(image_path)
        except IOError:
            print('path is not a valid image path. Please check: %s' % image_path)
            return
    elif islist(image_path):            # a list of images / a list of image paths
        imagelist = image_path
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
                visualize_image(image_tmp, vis=vis, save_path=save_path[i], save=save, debug=debug)
            else:
                visualize_image(image_tmp, vis=vis, debug=debug)
            index += 1
        return
    else:                               # an image
        if ispilimage(image_path):
            image = np.array(image_path)
        else:
            image = image_path

    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # debug=True
    # display image
    if iscolorimage(image, debug=debug):
        if is_cvimage:
            b,g,r = cv2.split(image)       # get b,g,r
            image = cv2.merge([r,g,b])     # switch it to rgb

        if warning:
            print 'visualizing color image'
        ax.imshow(image, interpolation='nearest')
    elif isgrayimage(image, debug=debug):
        if warning:
            print 'visualizing grayscale image'
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.reshape(image, image.shape[:-1])

        if isfloatimage(image, debug=debug) and all(item == 1.0 for item in image.flatten().tolist()):
            if warning:
                print('all elements in image are 1. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 0.00001
        elif isuintimage(image, debug=debug) and all(item == 255 for item in image.flatten().tolist()):
            if warning:
                print('all elements in image are 255. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 1
        ax.imshow(image, interpolation='nearest', cmap='gray')
    else:
        print 'unknown image type'
        # ax.imshow(image, interpolation='nearest')
        assert False, 'image is not correct'

    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_pts_covariance(pts_array, conf=None, std=None, fig=None, ax=None, debug=True, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        pts_array       : 2 x N numpy array of the data points.
        std            : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    if debug:
        assert is2dptsarray(pts_array), 'input points are not correct: (2, num_pts) vs %s' % print_np_shape(pts_array)
        if conf is not None:
            assert isscalar(conf) and conf >= 0 and conf <= 1, 'the confidence is not in a good range'
        if std is not None:
            assert ispositiveinteger(std), 'the number of standard deviation should be a positive integer'

    pts_array = np.transpose(pts_array)
    center = pts_array.mean(axis=0)
    covariance = np.cov(pts_array, rowvar=False)
    return visualize_covariance_ellipse(covariance=covariance, center=center, conf=conf, std=std, fig=fig, ax=ax, debug=debug, **kwargs), np.sqrt(covariance[0, 0]**2 + covariance[1, 1]**2)

def visualize_covariance_ellipse(covariance, center, conf=None, std=None, fig=None, ax=None, debug=True, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
        covariance      : The 2x2 covariance matrix to base the ellipse on
        center          : The location of the center of the ellipse. Expects a 2-element sequence of [x0, y0].
        conf            : a floating number between [0, 1]
        std             : The radius of the ellipse in numbers of standard deviations. Defaults to 2 standard deviations.
        ax              : The axis that the ellipse will be plotted on. Defaults to the current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
        A covariance ellipse
    """
    if debug:
        if conf is not None:
            assert isscalar(conf) and conf >= 0 and conf <= 1, 'the confidence is not in a good range'
        if std is not None:
            assert ispositiveinteger(std), 'the number of standard deviation should be a positive integer'

    def eigsorted(covariance):
        vals, vecs = np.linalg.eigh(covariance)
        # order = vals.argsort()[::-1]
        # return vals[order], vecs[:,order]
        return vals, vecs

    if conf is not None:
        conf = np.asarray(conf)
    elif std is not None:
        conf = 2 * norm.cdf(std) - 1
    else:
        raise ValueError('One of `conf` and `std` should be specified.')
    r2 = chi2.ppf(conf, 2)

    fig, ax = get_fig_ax_helper(fig=fig, ax=ax)

    vals, vecs = eigsorted(covariance)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # theta = np.degrees(np.arctan2(*vecs[::-1, 0]))

    # Width and height are "full" widths, not radius
    # width, height = 2 * std * np.sqrt(vals)
    width, height = 2 * np.sqrt(np.sqrt(vals) * r2)
    # width, height = 2 * np.sqrt(vals[:, None] * r2)

    ellipse = Ellipse(xy=center, width=width, height=height, angle=theta, **kwargs)
    ellipse.set_facecolor('none')

    ax.add_artist(ellipse)
    return ellipse

def visualize_pts_array(pts_array, covariance=False, color_index=0, fig=None, ax=None, pts_size=20, label=False, label_list=None, label_size=2, xlim=None, ylim=None, occlusion=True, vis_threshold=-10000, save_path=None, vis=False, debug=True, warning=False, closefig=True):
    '''
    plot keypoints

    parameters:
        pts_array:      2 or 3 x num_pts, the third channel could be confidence or occlusion
    '''

    fig, ax = get_fig_ax_helper(fig=fig, ax=ax)
    std = None
    conf = 0.95
    if islist(color_index):
        if debug:
            assert not occlusion, 'the occlusion is not compatible with plotting different colors during scattering'
            assert not covariance, 'the covariance is not compatible with plotting different colors during scattering'
        color_tmp = [color_set_big[index_tmp] for index_tmp in color_index]
    else:
        color_tmp = color_set_big[color_index % len(color_set_big)]
    num_pts = pts_array.shape[1]

    if is2dptsarray(pts_array):    
        ax.scatter(pts_array[0, :], pts_array[1, :], color=color_tmp, s=pts_size)

        if debug and islist(color_tmp):
            assert len(color_tmp) == pts_array.shape[1], 'number of points to plot is not equal to number of colors provided'
        pts_visible_index = range(pts_array.shape[1])
        pts_ignore_index = []
        pts_invisible_index = []
    else:
        num_float_elements = np.where(np.logical_and(pts_array[2, :] > 0, pts_array[2, :] < 1))[0].tolist()
        if len(num_float_elements) > 0:
            type_3row = 'conf'
            if warning:
                print('third row is confidence')
        else:
            type_3row = 'occu'
            if warning:
                print('third row is occlusion')

        if type_3row == 'occu':
            pts_visible_index   = np.where(pts_array[2, :] == 1)[0].tolist()              # plot visible points in red color
            pts_invisible_index = np.where(pts_array[2, :] == 0)[0].tolist()              # plot invisible points in blue color
            pts_ignore_index    = np.where(pts_array[2, :] == -1)[0].tolist()             # do not plot points with annotation
        else:
            pts_visible_index   = np.where(pts_array[2, :] > vis_threshold)[0].tolist()
            pts_ignore_index    = np.where(pts_array[2, :] <= vis_threshold)[0].tolist()
            pts_invisible_index = []

        if debug and islist(color_tmp):
            assert len(color_tmp) == len(pts_visible_index), 'number of points to plot is not equal to number of colors provided'

        ax.scatter(pts_array[0, pts_visible_index], pts_array[1, pts_visible_index], color=color_tmp, s=pts_size)
        if occlusion:
            ax.scatter(pts_array[0, pts_invisible_index], pts_array[1, pts_invisible_index], color=color_set_big[(color_index+1) % len(color_set_big)], s=pts_size)
        # else:
            # ax.scatter(pts_array[0, pts_invisible_index], pts_array[1, pts_invisible_index], color=color_tmp, s=pts_size)
        if covariance:
            visualize_pts_covariance(pts_array[0:2, :], std=std, conf=conf, fig=fig, ax=ax, debug=False, color=color_tmp)

    if label:
        for pts_index in xrange(num_pts):
            label_tmp = label_list[pts_index]
            # print label_tmp
            if pts_index in pts_ignore_index:
                continue
            else:
                # note that the annotation is based on the coordinate instead of the order of plotting the points, so the orider in pts_index does not matter
                if islist(color_index):
                    plt.annotate(label_tmp, xy=(pts_array[0, pts_index], pts_array[1, pts_index]), xytext=(-1, 1), color=color_set_big[(color_index[pts_index]+5) % len(color_set_big)], textcoords='offset points', ha='right', va='bottom', fontsize=label_size)
                else:
                    plt.annotate(label_tmp, xy=(pts_array[0, pts_index], pts_array[1, pts_index]), xytext=(-1, 1), color=color_set_big[(color_index+5) % len(color_set_big)], textcoords='offset points', ha='right', va='bottom', fontsize=label_size)
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                # arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
    if xlim is not None:
        if debug:
            assert islist(xlim) and len(xlim) == 2, 'the x lim is not correct'
        plt.xlim(xlim[0], xlim[1])

    if ylim is not None:    
        if debug:
            assert islist(ylim) and len(ylim) == 2, 'the y lim is not correct'
        plt.ylim(ylim[0], ylim[1])

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig, transparent=False)

def visualize_image_with_pts(image_path, pts, pts_size=20, label=False, label_list=None, label_size=20, color_index=0, is_cvimage=False, vis_threshold=-10000, vis=False, save_path=None, debug=True, closefig=True):
    '''
    visualize image and plot keypoints on top of it

    parameter:
        image_path:     a path to an image / an image
        pts:            a dictionary or 2 or 3 x num_pts numpy array
                        when there are 3 channels in pts, the third one denotes the occlusion flag
                        occlusion: 0 -> invisible, 1 -> visible, -1 -> not existing
        label:          determine to add text label for each point
        label_list:     label string for all points
        color_index:    a scalar or a list of color indexes
    '''
    fig, ax = visualize_image(image_path, is_cvimage=is_cvimage, vis=False, debug=False, closefig=False)

    if label and (label_list is None):
        if not isdict(pts):
            num_pts = pts.shape[1]
        else:
            pts_tmp = pts.values()[0]
            num_pts = np.asarray(pts_tmp).shape[1] if islist(pts_tmp) else pts_tmp.shape[1]
        label_list = [str(i+1) for i in xrange(num_pts)]
    # print label_list

    if debug:
        assert not islist(image_path), 'this function only support to plot points on one image'
        if isdict(pts):
            for pts_tmp in pts.values():
                if islist(pts_tmp):
                    pts_tmp = np.asarray(pts_tmp)
                assert is2dptsarray(pts_tmp) or is2dptsarray_occlusion(pts_tmp), 'input points within dictionary are not correct: (2 (3), num_pts) vs %s' % print_np_shape(pts_tmp)
        else:
            assert is2dptsarray(pts) or is2dptsarray_occlusion(pts), 'input points are not correct'
        assert islogical(label), 'label flag is not correct'
        if label:
            assert islist(label_list) and all(isstring(label_tmp) for label_tmp in label_list), 'labels are not correct'

    if isdict(pts):
        color_index = color_index
        for pts_id, pts_array in pts.items():
            if islist(pts_array):
                pts_array = np.asarray(pts_array)
            visualize_pts_array(pts_array, fig=fig, ax=ax, color_index=color_index, pts_size=pts_size, label=label, label_list=label_list, label_size=label_size, occlusion=True, vis_threshold=vis_threshold, debug=debug, closefig=False)
            color_index += 1
    else:   
        visualize_pts_array(pts, fig=fig, ax=ax, color_index=color_index, pts_size=pts_size, label=label, label_list=label_list, label_size=label_size, occlusion=True, vis_threshold=vis_threshold, debug=debug, closefig=False)

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_bbox(bbox, fig=None, ax=None, linewidth=0.5, color_index=20, vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize a set of bounding box

    parameters:
        bbox:       N x 4
    '''
    if debug:    
        assert bboxcheck_TLBR(bbox), 'input bounding boxes are not correct'

    edge_color = color_set_big[color_index % len(color_set_big)]

    # plot bounding box
    bbox = bbox_TLBR2TLWH(bbox, debug=debug)              # convert TLBR format to TLWH format
    for bbox_index in range(bbox.shape[0]):
        bbox_tmp = bbox[bbox_index, :]     
        ax.add_patch(plt.Rectangle((bbox_tmp[0], bbox_tmp[1]), bbox_tmp[2], bbox_tmp[3], fill=False, edgecolor=edge_color, linewidth=linewidth))

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_image_with_bbox(image, bbox, ax=None, vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize image and plot keypoints on top of it

    parameter:
        image:          a path to an image / an image
        bbox:           N X 4 numpy array, with TLBR format
    '''
    if debug:
        assert not islist(image), 'this function only support to plot points on one image'

    fig, ax = visualize_image(image, vis=False, debug=debug, closefig=False)
    return visualize_bbox(bbox, fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_image_with_pts_bbox(image, pts_array, window_size, pts_size=20, label=False, label_list=None, color_index=0, vis=False, save_path=None, debug=True, closefig=True):
    '''
    plot a set of points on top of an image with bbox around all points

    parameters
        pts_array:              2 x N
    '''
    if debug:
        assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'input points are not correct'

    fig, ax = visualize_image_with_pts(image, pts_array, pts_size=pts_size, label=label, label_list=label_list, color_index=color_index, debug=False, save_path=None, closefig=False)
    bbox = get_centered_bbox(pts_array, window_size, window_size, debug=debug)
    return visualize_bbox(bbox, fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_image_with_pts_bbox_tracking(image, pts_array, valid_index, window_size, pts_anno=None, pts_size=20, vis=False, save_path=None, debug=True, closefig=True):
    '''
    plot a set of points from tracking results on top of an image with bbox, and plot the annotation meanwhile with another color
    the tracking results also include the successful or failed tracking, we differentiate them in different color

    parameters:
        pts_array:              2 x N
        pts_anno:               2 x N
        valid_index:            a list of m elements who succeeds, m >= 0 && m <= N
    '''
    if debug:
        assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'input points are not correct'
        if pts_anno is not None:
            assert pts_array.shape == pts_anno.shape, 'the input points from prediction and annotation have to have the same shape'
        assert islist(valid_index), 'the valid index is not a list'

    num_pts_all = pts_array.shape[1]
    num_pts_succeed = len(valid_index)
    if debug:
        assert num_pts_succeed <= num_pts_all, 'the number of points should be less than number of points in total'

    color_anno_index = 16                         # cyan
    color_succeed_index = 15                       # yellow
    color_failed_index = 0                        # aqua
    failed_index = remove_list_from_list(range(num_pts_all), valid_index, debug=debug)
    pts_succeed = pts_array[:, valid_index]
    pts_failed = pts_array[:, failed_index]

    # plot successful predictions
    fig, ax = visualize_image_with_pts(image, pts_succeed, pts_size=pts_size, color_index=color_succeed_index, debug=False, closefig=False)
    bbox = get_centered_bbox(pts_succeed, window_size, window_size, debug=debug)
    fig, ax = visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_succeed_index, vis=vis, debug=debug, closefig=False)

    # plot failed predictions
    fig, ax = visualize_pts_array(pts_failed, fig=fig, ax=ax, color_index=color_failed_index, pts_size=pts_size, debug=debug, closefig=False)
    bbox = get_centered_bbox(pts_failed, window_size, window_size, debug=debug)
    
    if pts_anno is None:
        return visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_failed_index, vis=vis, save_path=save_path, debug=debug, closefig=closefig)    
    else:
        fig, ax = visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_failed_index, vis=vis, debug=debug, closefig=False)    

        # plot annotations
        fig, ax = visualize_pts_array(pts_anno, fig=fig, ax=ax, color_index=color_anno_index, pts_size=pts_size, debug=debug, closefig=False)
        bbox = get_centered_bbox(pts_anno, window_size, window_size, debug=debug)
        return visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_anno_index, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_lines(lines_array, color_index=0, line_width=3, fig=None, ax=None, vis=True, save_path=None, debug=True, closefig=True):
    '''
    plot lines 

    parameters:
        lines_array:            4 x num_lines, each column denotes (x1, y1, x2, y2)
    '''

    if debug:    
        assert islinesarray(lines_array), 'input array of lines are not correct'

    fig, ax = get_fig_ax_helper(fig=fig, ax=ax)

    # plot lines
    num_lines = lines_array.shape[1]
    lines_all = []
    for line_index in range(num_lines):
        line_tmp = lines_array[:, line_index]
        lines_all.append([tuple([line_tmp[0], line_tmp[1]]), tuple([line_tmp[2], line_tmp[3]])])

    line_col = plycollections.LineCollection(lines_all, linewidths=line_width, colors=color_set[color_index])
    ax.add_collection(line_col)
        # ax.plot([line_tmp[0], line_tmp[2]], [line_tmp[1], line_tmp[3]], color=color_set[color_index], linewidth=line_width)

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_pts_line(pts_array, line_index_list, method=2, threshold=None, pts_size=20, line_size=10, line_color_index=5, fig=None, ax=None, vis=False, save_path=None, closefig=True, debug=True, seed=0, alpha=0.5):
    '''
    given a list of index, and a point array, to plot a set of points with line on it

    inputs:
        pts_array:          2(3) x num_pts
        line_index_list:    a list of index
        method:             1: all points are connected, if some points are missing in the middle, just ignore that point and connect the two nearby points
                            2: if some points are missing, there might be two sub-lines
        threshold:          confidence to draw the points

    '''

    if debug:
        assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'input points are not correct'
        assert islist(line_index_list), 'the list of index is not correct'
        assert method == 2 or method == 1, 'the plot method is not correct'

    num_pts = pts_array.shape[1]
    # expand the pts_array to 3 rows if the confidence row is not provided
    if pts_array.shape[0] == 2:
        pts_array = np.vstack((pts_array, np.ones((1, num_pts))))

    fig, ax = get_fig_ax_helper(fig=fig, ax=ax)
    np.random.seed(seed)
    color_option = 'hsv'

    if color_option == 'rgb':
        color_set_random = np.random.rand(3, num_pts)
    elif color_option == 'hsv':
        h_random = np.random.rand(num_pts, )
        color_set_random = np.zeros((3, num_pts), dtype='float32')
        for pts_index in range(num_pts):
            # print(h_random[pts_index])
            # print(colorsys.hsv_to_rgb(h_random[pts_index], 1, 1))
            color_set_random[:, pts_index] = colorsys.hsv_to_rgb(h_random[pts_index], 1, 1) 

    line_color = color_set[line_color_index]
    pts_line = pts_array[:, line_index_list]

    if method == 1:    
        valid_pts_list = np.where(pts_line[2, :] > threshold)[0].tolist()
        pts_line_tmp = pts_line[:, valid_pts_list]
        ax.plot(pts_line_tmp[0, :], pts_line_tmp[1, :], lw=line_size, color=line_color, alpha=alpha)      # plot all lines

        # plot all points
        for pts_index in valid_pts_list:
            pts_index_original = line_index_list[pts_index]
            # ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], 'o', color=color_set_big[pts_index_original % len(color_set_big)], alpha=alpha)
            ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], marker='o', ms=pts_size, lw=line_size, color=color_set_random[:, pts_index], alpha=alpha)
    else:
        not_valid_pts_list = np.where(pts_line[2, :] < threshold)[0].tolist()
        if len(not_valid_pts_list) == 0:            # all valid
            ax.plot(pts_line[0, :], pts_line[1, :], lw=line_size, color=line_color, alpha=alpha)

            # plot points
            for pts_index in line_index_list:
                # ax.plot(pts_array[0, pts_index], pts_array[1, pts_index], 'o', color=color_set_big[pts_index % len(color_set_big)], alpha=alpha)
                ax.plot(pts_array[0, pts_index], pts_array[1, pts_index], marker='o', ms=pts_size, lw=line_size, color=color_set_random[:, pts_index], alpha=alpha)
        else:
            prev_index = 0
            for not_valid_index in not_valid_pts_list:
                plot_list = range(prev_index, not_valid_index)
                pts_line_tmp = pts_line[:, plot_list]
                ax.plot(pts_line_tmp[0, :], pts_line_tmp[1, :], lw=line_size, color=line_color, alpha=alpha)
                
                # plot points
                for pts_index in plot_list:
                    pts_index_original = line_index_list[pts_index]
                    ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], marker='o', ms=pts_size, lw=line_size, color=color_set_random[:, pts_index_original], alpha=alpha) 
                    # ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], 'o', color=color_set_big[pts_index_original % len(color_set_big)], alpha=alpha) 

                prev_index = not_valid_index + 1

            pts_line_tmp = pts_line[:, prev_index:]
            ax.plot(pts_line_tmp[0, :], pts_line_tmp[1, :], lw=line_size, color=line_color, alpha=alpha)      # plot last line

            # plot last points
            for pts_index in range(prev_index, pts_line.shape[1]):
                pts_index_original = line_index_list[pts_index]
                # ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], 'o', color=color_set_big[pts_index_original % len(color_set_big)], alpha=alpha) 
                ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], marker='o', ms=pts_size, lw=line_size, color=color_set_random[:, pts_index_original], alpha=alpha) 

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_pts(pts, title=None, fig=None, ax=None, display_range=False, xlim=[-100, 100], ylim=[-100, 100], display_list=None, covariance=False, mse=False, mse_value=None, vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize point scatter plot

    parameter:
        pts:            2 x num_pts numpy array or a dictionary containing 2 x num_pts numpy array
    '''

    if debug:
        if isdict(pts):
            for pts_tmp in pts.values():
                assert is2dptsarray(pts_tmp) , 'input points within dictionary are not correct: (2, num_pts) vs %s' % print_np_shape(pts_tmp)
            if display_list is not None:
                assert islist(display_list) and len(display_list) == len(pts), 'the input display list is not correct'
                assert CHECK_EQ_LIST_UNORDERED(display_list, pts.keys(), debug=debug), 'the input display list does not match the points key list'
            else:
                display_list = pts.keys()
        else:
            assert is2dptsarray(pts), 'input points are not correct: (2, num_pts) vs %s' % print_np_shape(pts)
        if title is not None:
            assert isstring(title), 'title is not correct'
        else:
            title = 'Point Error Vector Distribution Map'
        assert islogical(display_range), 'the flag determine if to display in a specific range should be logical value'
        if display_range:
            assert islist(xlim) and islist(ylim) and len(xlim) == 2 and len(ylim) == 2, 'the input range for x and y is not correct'
            assert xlim[1] > xlim[0] and ylim[1] > ylim[0], 'the input range for x and y is not correct'

    # figure setting
    width = 1024
    height = 1024
    figsize = width / float(dpi), height / float(dpi)
    if fig is None:
        fig = plt.figure(figsize=figsize)

    if ax is None:
        plt.title(title, fontsize=20)

        if isdict(pts):
            num_pts_all = pts.values()[0].shape[1]
            if all(pts_tmp.shape[1] == num_pts_all for pts_tmp in pts.values()):
                plt.xlabel('x coordinate (%d points)' % pts.values()[0].shape[1], fontsize=16)
                plt.ylabel('y coordinate (%d points)' % pts.values()[0].shape[1], fontsize=16)
            else:
                print('number of points is different across different methods')
                plt.xlabel('x coordinate', fontsize=16)
                plt.ylabel('y coordinate', fontsize=16)
        else:
            plt.xlabel('x coordinate (%d points)' % pts.shape[1], fontsize=16)
            plt.ylabel('y coordinate (%d points)' % pts.shape[1], fontsize=16)
        plt.axis('equal')
        ax = plt.gca()
        ax.grid()
    
    # internal parameters
    pts_size = 5
    std = None
    conf = 0.98
    color_index = 0
    marker_index = 0
    hatch_index = 0
    alpha = 0.2
    legend_fontsize = 10
    scale_distance = 48.8
    linewidth = 2

    # plot points
    handle_dict = dict()    # for legend
    if isdict(pts):
        num_methods = len(pts)
        assert len(color_set) * len(marker_set) >= num_methods and len(color_set) * len(hatch_set) >= num_methods, 'color in color set is not enough to use, please use different markers'
        mse_return = dict()
        for method_name, pts_tmp in pts.items():
            color_tmp = color_set[color_index]
            marker_tmp = marker_set[marker_index]
            hatch_tmp = hatch_set[hatch_index]

            # plot covariance ellipse
            if covariance:
                _, covariance_number = visualize_pts_covariance(pts_tmp[0:2, :], std=std, conf=conf, ax=ax, debug=debug, color=color_tmp, hatch=hatch_tmp, linewidth=linewidth)
            
            handle_tmp = ax.scatter(pts_tmp[0, :], pts_tmp[1, :], color=color_tmp, marker=marker_tmp, s=pts_size, alpha=alpha)    
            if mse:
                if mse_value is None:
                    num_pts = pts_tmp.shape[1]
                    mse_tmp, _ = pts_euclidean(pts_tmp[0:2, :], np.zeros((2, num_pts), dtype='float32'), debug=debug)
                else:
                    mse_tmp = mse_value[method_name]
                display_string = '%s, MSE: %.1f (%.1f um), Covariance: %.1f' % (method_name, mse_tmp, mse_tmp * scale_distance, covariance_number)
                mse_return[method_name] = mse_tmp
            else:
                display_string = method_name
            handle_dict[display_string] = handle_tmp
            color_index += 1
            if color_index / len(color_set) == 1:            
                marker_index += 1
                hatch_index += 1
                color_index = color_index % len(color_set)

        # reorder the handle before plot
        handle_key_list = handle_dict.keys()
        handle_value_list = handle_dict.values()
        order_index_list = [display_list.index(method_name_tmp.split(', ')[0]) for method_name_tmp in handle_dict.keys()]
        ordered_handle_key_list = list_reorder(handle_key_list, order_index_list, debug=debug)
        ordered_handle_value_list = list_reorder(handle_value_list, order_index_list, debug=debug)
        plt.legend(list2tuple(ordered_handle_value_list), list2tuple(ordered_handle_key_list), scatterpoints=1, markerscale=4, loc='lower left', fontsize=legend_fontsize)
        
    else:
        color_tmp = color_set[color_index]
        marker_tmp = marker_set[marker_index]
        hatch_tmp = hatch_set[hatch_index]

        handle_tmp = ax.scatter(pts[0, :], pts[1, :], color=color_tmp, marker=marker_tmp, s=pts_size, alpha=alpha)    

        # plot covariance ellipse
        if covariance:
            _, covariance_number = visualize_pts_covariance(pts[0:2, :], std=std, conf=conf, ax=ax, debug=debug, color=color_tmp, hatch=hatch_tmp, linewidth=linewidth)

        if mse:
            if mse_value is None:
                num_pts = pts.shape[1]
                mse_tmp, _ = pts_euclidean(pts[0:2, :], np.zeros((2, num_pts), dtype='float32'), debug=debug)
                display_string = 'MSE: %.1f (%.1f um), Covariance: %.1f' % (mse_tmp, mse_tmp * scale_distance, covariance_number)
                mse_return = mse_tmp
            else:
                display_string = 'MSE: %.1f (%.1f um), Covariance: %.1f' % (mse_value, mse_value * scale_distance, covariance_number)
                mse_return = mse_value
            handle_dict[display_string] = handle_tmp
            plt.legend(list2tuple(handle_dict.values()), list2tuple(handle_dict.keys()), scatterpoints=1, markerscale=4, loc='lower left', fontsize=legend_fontsize)
            
    # display only specific range
    if display_range:
        axis_bin = 10 * 2
        interval_x = (xlim[1] - xlim[0]) / axis_bin
        interval_y = (ylim[1] - ylim[0]) / axis_bin
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks(np.arange(xlim[0], xlim[1] + interval_x, interval_x))
        plt.yticks(np.arange(ylim[0], ylim[1] + interval_y, interval_y))
    plt.grid()

    save_vis_close_helper(fig=fig, ax=ax, vis=vis, transparent=False, save_path=save_path, debug=debug, closefig=closefig)
    return mse_return

def visualize_ced(normed_mean_error_dict, error_threshold, normalized=True, truncated_list=None, display2terminal=True, display_list=None, title=None, fig=None, ax=None, debug=True, vis=True, pck_savepath=None, table_savepath=None, closefig=True):
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
        assert islogical(normalized), 'the normalization flag should be logical'
        if normalized:
            assert error_threshold > 0 and error_threshold < 100, 'threshold percentage is not well set'
        if save:
            assert pck_savepath is not None and is_path_exists_or_creatable(pck_savepath), 'please provide a valid path to save the pck results' 
            assert table_savepath is not None and is_path_exists_or_creatable(table_savepath), 'please provide a valid path to save the table results' 
        if title is not None:
            assert isstring(title), 'title is not correct'
        else:
            title = '2D PCK curve'
        if truncated_list is not None:
            assert islist(truncated_list) and all(isscalar(thres_tmp) for thres_tmp in truncated_list), 'the input truncated list is not correct'
        if display_list is not None:
            assert islist(display_list) and len(display_list) == len(normed_mean_error_dict), 'the input display list is not correct'
            assert CHECK_EQ_LIST_UNORDERED(display_list, normed_mean_error_dict.keys(), debug=debug), 'the input display list does not match the error dictionary key list'
        else:
            display_list = normed_mean_error_dict.keys()

    # set display parameters
    width = 1000
    height = 800
    legend_fontsize = 10
    scale_distance = 48.8
    line_index = 0
    color_index = 0

    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)

    # set figure handle
    num_bins = 1000
    if normalized:
        maximum_x = 1
        scale = num_bins / 100
    else:
        maximum_x = error_threshold + 1
        scale = num_bins / maximum_x        
    x_axis = np.linspace(0, maximum_x, num_bins)            # error axis, percentage of normalization factor    
    y_axis = np.zeros(num_bins)
    interval_y = 10
    interval_x = 1
    plt.xlim(0, error_threshold)
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.xticks(np.arange(0, error_threshold + interval_x, interval_x))
    plt.grid()
    plt.title(title, fontsize=20)
    if normalized:
        plt.xlabel('Normalized error euclidean distance (%)', fontsize=16)
    else:
        plt.xlabel('Absolute error euclidean distance', fontsize=16)

    # calculate metrics for each method
    num_methods = len(normed_mean_error_dict)
    num_images = len(normed_mean_error_dict.values()[0])
    metrics_dict = dict()
    metrics_table = list()
    table_title = ['Method Name / Metrics', 'AUC', 'MSE']
    append2title = False
    assert num_images > 0, 'number of error array should be larger than 0'
    for ordered_index in range(num_methods):
        method_name = display_list[ordered_index]
        normed_mean_error = normed_mean_error_dict[method_name]

        if debug:
            assert isnparray(normed_mean_error) and normed_mean_error.ndim == 1, 'shape of error distance is not good'
            assert len(normed_mean_error) == num_images, 'number of testing images should be equal for all methods'
            assert len(linestyle_set) * len(color_set) >= len(normed_mean_error_dict)

        color_tmp = color_set[color_index]
        line_tmp = linestyle_set[line_index]
        
        for i in range(num_bins):
            y_axis[i] = float((normed_mean_error < x_axis[i]).sum()) / num_images                       # percentage of error

        # calculate area under the curve and mean square error
        entry = dict()
        entry['AUC'] = np.sum(y_axis[:error_threshold * scale]) / (error_threshold * scale)         # bigger, better
        entry['MSE'] = np.mean(normed_mean_error)                                                                      # smaller, better
        metrics_table_tmp = [str(method_name), '%.2f' % (entry['AUC']), '%.1f' % (entry['MSE'])]
        if truncated_list is not None:
            tmse_dict = calculate_truncated_mse(normed_mean_error.tolist(), truncated_list, debug=debug)
            for threshold in truncated_list:
                entry['AUC/%s'%threshold] = np.sum(y_axis[:error_threshold * scale]) / (error_threshold * scale)         # bigger, better
                entry['MSE/%s'%threshold] = tmse_dict[threshold]['T-MSE']
                entry['percentage/%s'%threshold] = tmse_dict[threshold]['percentage']
                
                if not append2title:
                    table_title.append('AUC/%s'%threshold)
                    table_title.append('MSE/%s'%threshold)
                    table_title.append('pct/%s'%threshold)
                metrics_table_tmp.append('%.2f' % (entry['AUC/%s'%threshold]))
                metrics_table_tmp.append('%.1f' % (entry['MSE/%s'%threshold]))
                metrics_table_tmp.append('%.1f' % (100 * entry['percentage/%s'%threshold]) + '%')

        # print metrics_table_tmp
        metrics_table.append(metrics_table_tmp)
        append2title = True
        metrics_dict[method_name] = entry

        # draw 
        label = '%s, AUC: %.2f, MSE: %.1f (%.0f um)' % (method_name, entry['AUC'], entry['MSE'], entry['MSE'] * scale_distance)
        # print label
        if normalized:
            plt.plot(x_axis*100, y_axis*100, color=color_tmp, linestyle=line_tmp, label=label, lw=3)
        else:
            plt.plot(x_axis, y_axis*100, color=color_tmp, linestyle=line_tmp, label=label, lw=3)
        plt.legend(loc=4, fontsize=legend_fontsize)
        
        color_index += 1
        if color_index / len(color_set) == 1:            
            line_index += 1
            color_index = color_index % len(color_set)

    plt.grid()
    plt.ylabel('{} Test Images (%)'.format(num_images), fontsize=16)

    save_vis_close_helper(fig=fig, ax=ax, vis=vis, transparent=False, save_path=pck_savepath, debug=debug, closefig=closefig)
    # if vis:
        # plt.show()
    # if save:
        # fig.savefig(pck_savepath, dpi=dpi)
        # if display2terminal:
            # print 'save PCK curve to %s' % pck_savepath
    # plt.close(fig)

    # reorder the table
    order_index_list = [display_list.index(method_name_tmp) for method_name_tmp in normed_mean_error_dict.keys()]
    order_index_list = [0] + [order_index_tmp + 1 for order_index_tmp in order_index_list]

    # print table to terminal
    metrics_table = [table_title] + metrics_table
    # metrics_table = list_reorder([table_title] + metrics_table, order_index_list, debug=debug)
    table = AsciiTable(metrics_table)
    if display2terminal:
        print '\nprint detailed metrics'
        print table.table
        
    # save table to file
    if table_savepath is not None:
        table_file = open(table_savepath, 'w')
        table_file.write(table.table)
        table_file.close()
        if display2terminal:
            print '\nsave detailed metrics to %s' % table_savepath
    
    return metrics_dict, metrics_table


def visualize_nearest_neighbor(featuremap_dict, num_neighbor=5, top_number=5, vis=True, save_csv=False, csv_save_path=None, save_vis=False, save_img=False, save_thumb_name='nearest_neighbor.png', img_src_folder=None, ext_filter='.jpg', nn_save_folder=None, debug=True):
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


def visualize_distribution(data, bin_size=None, vis=False, save_path=None, debug=True, closefig=True):
    '''
    visualize the histogram of a data, which can be a dictionary or list or numpy array or tuple or a list of list
    '''
    if debug:
        assert istuple(data) or isdict(data) or islist(data) or isnparray(data), 'input data is not correct'

    # convert data type
    if istuple(data):
        data = list(data)
    elif isdict(data):
        data = data.values()
    elif isnparray(data):
        data = data.tolist()

    num_bins = 1000.0
    fig, ax = get_fig_ax_helper(fig=None, ax=None)

    # calculate bin size
    if bin_size is None:
        if islistoflist(data):
            max_value = np.max(np.max(data))
            min_value = np.min(np.min(data))
        else:
            max_value = np.max(data)
            min_value = np.min(data)
        bin_size = (max_value - min_value) / num_bins
    else:
        try:
            bin_size = float(bin_size)
        except TypeError:
            print('size of bin should be an float value')

    # plot
    if islistoflist(data):
        max_value = np.max(np.max(data))
        min_value = np.min(np.min(data))
        bins = np.arange(min_value - bin_size, max_value + bin_size, bin_size)      # fixed bin size
        plt.xlim([min_value - bin_size, max_value + bin_size])
        for data_list_tmp in data:
            if debug:
                assert islist(data_list_tmp), 'the nested list is not correct!'
            # plt.hist(data_list_tmp, bins=bins, alpha=0.3)
            sns.distplot(data_list_tmp, bins=bins, kde=False)
            # sns.distplot(data_list_tmp, bins=bins, kde=False)
    else:
        bins = np.arange(min(data) - 10 * bin_size, max(data) + 10 * bin_size, bin_size)      # fixed bin size
        plt.xlim([min(data) - bin_size, max(data) + bin_size])
        plt.hist(data, bins=bins, alpha=0.5)
    
    plt.title('distribution of data')
    plt.xlabel('data (bin size = %f)' % bin_size)
    plt.ylabel('count')

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_bar(data, bin_size=2.0, title='Bar Graph of Key-Value Pair', xlabel='index', ylabel='count', vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize the bar graph of a data, which can be a dictionary or list of dictionary

    different from function of visualize_bar_graph, this function does not depend on panda and dataframe, it's simpler but with less functionality
    also the key of this function takes continuous scalar variable
    '''
    if debug:
        assert isstring(title) and isstring(xlabel) and isstring(ylabel), 'title/xlabel/ylabel is not correct'
        assert isdict(data) or islist(data), 'input data is not correct'
        assert isscalar(bin_size), 'the bin size is not a floating number'

    if isdict(data):
        index_list = data.keys()
        if debug:
            assert islistofscalar(index_list), 'the input dictionary does not contain a scalar key'
        frequencies = data.values()
    else:
        index_list = range(len(data))
        frequencies = data

    index_str_list = scalar_list2str_list(index_list, debug=debug)
    index_list = np.array(index_list)
    fig, ax = get_fig_ax_helper(fig=None, ax=None)
    # ax.set_xticks(index_list)
    # ax.set_xticklabels(index_str_list)
    plt.bar(index_list, frequencies, bin_size, color='r', alpha=0.5)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, transparent=False, closefig=closefig)

def visualize_bar_graph(data, title='Bar Graph of Key-Value Pair', xlabel='pixel error', ylabel='keypoint index', label=False, label_list=None, vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize the bar graph of a data, which can be a dictionary or list of dictionary
    inside each dictionary, the keys (string) should be the same which is the y label, the values should be scalar
    '''
    if debug:
        assert isstring(title) and isstring(xlabel) and isstring(ylabel), 'title/xlabel/ylabel is not correct'
        assert isdict(data) or islistofdict(data), 'input data is not correct'
        if isdict(data):
            assert all(isstring(key_tmp) for key_tmp in data.keys()), 'the keys are not all strings'
            assert all(isscalar(value_tmp) for value_tmp in data.values()), 'the keys are not all strings'
        else:
            assert len(data) <= len(color_set), 'number of data set is larger than number of color to use'
            keys = sorted(data[0].keys())
            for dict_tmp in data:
                if not (sorted(dict_tmp.keys()) == keys):
                    print dict_tmp.keys()
                    print keys
                    assert False, 'the keys are not equal across different input set'
                assert all(isstring(key_tmp) for key_tmp in dict_tmp.keys()), 'the keys are not all strings'
                assert all(isscalar(value_tmp) for value_tmp in dict_tmp.values()), 'the values are not all scalars'   

    # convert dictionary to DataFrame
    data_new = dict()
    if isdict(data):
        key_list = data.keys()
        sorted_index = sorted(range(len(key_list)), key=lambda k: key_list[k])
        data_new['names'] = (np.asarray(key_list)[sorted_index]).tolist()
        data_new['values'] = (np.asarray(data.values())[sorted_index]).tolist()
    else:
        key_list = data[0].keys()
        sorted_index = sorted(range(len(key_list)), key=lambda k: key_list[k])
        data_new['names'] = (np.asarray(key_list)[sorted_index]).tolist()
        num_sets = len(data)        
        for set_index in range(num_sets):
            data_new['value_%03d'%set_index] = (np.asarray(data[set_index].values())[sorted_index]).tolist()
    dataframe = DataFrame(data_new)

    # plot
    width = 2000
    height = 2000
    alpha = 0.5
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    sns.set(style='whitegrid')
    # fig, ax = get_fig_ax_helper(fig=None, ax=None)
    if isdict(data):
        g = sns.barplot(x='values', y='names', data=dataframe, label='data', color='b')
        plt.legend(ncol=1, loc='lower right', frameon=True, fontsize=5)
    else:
        num_sets = len(data)
        for set_index in range(num_sets):
            if set_index == 0:
                sns.set_color_codes('pastel')
            else:
                sns.set_color_codes('muted')

            if label:
                sns.barplot(x='value_%03d'%set_index, y='names', data=dataframe, label=label_list[set_index], color=color_set[set_index], alpha=alpha)
            else:
                sns.barplot(x='value_%03d'%set_index, y='names', data=dataframe, color=solor_set[set_index], alpha=alpha)
        plt.legend(ncol=len(data), loc='lower right', frameon=True, fontsize=5)

    sns.despine(left=True, bottom=True)
    plt.title(title, fontsize=20)
    plt.xlim([0, 50])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    num_yticks = len(data_new['names'])
    adaptive_fontsize = -0.0555556 * num_yticks + 15.111
    plt.yticks(fontsize=adaptive_fontsize)

    return save_vis_close_helper(fig=fig, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

###################################################################################################################################################### helper
def get_fig_ax_helper(fig=None, ax=None):
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()   

    return fig, ax

def save_vis_close_helper(fig=None, ax=None, vis=False, save_path=None, debug=True, transparent=True, closefig=True):
    # save and visualization
    if save_path is not None:
        if debug:
            assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not valid: %s' % save_path
            mkdir_if_missing(save_path)
        fig.savefig(save_path, dpi=dpi, transparent=transparent)
    if vis:
        plt.show()

    if closefig:
        plt.close(fig)
        return None, None
    else:
        return fig, ax

def autopct_generator(upper_percentage_to_draw):
    '''
    this function generate a autopct when draw a pie chart
    '''
    def inner_autopct(pct):
        return ('%.2f' % pct) if pct > upper_percentage_to_draw else ''
    return inner_autopct

def fixOverLappingText(text):
    # if undetected overlaps reduce sigFigures to 1
    sigFigures = 2
    positions = [(round(item.get_position()[1],sigFigures), item) for item in text]

    overLapping = Counter((item[0] for item in positions))
    overLapping = [key for key, value in overLapping.items() if value >= 2]

    for key in overLapping:
        textObjects = [text for position, text in positions if position == key]

        if textObjects:

            # If bigger font size scale will need increasing
            scale = 0.1

            spacings = np.linspace(0,scale*len(textObjects),len(textObjects))

            for shift, textObject in zip(spacings,textObjects):
                textObject.set_y(key + shift)