# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions for visualizing images 
import matplotlib.pyplot as plt
# import matplotlib as mpl; mpl.use('Agg')
# import warnings
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# from warnings import catch_warnings, simplefilter
# with catch_warnings(record=True):
    # simplefilter('ignore', FutureWarning)

from private import save_vis_close_helper, get_fig_ax_helper
from geometry_vis import visualize_pts_array
from xinshuo_images.python.private import safe_image
from xinshuo_miscellaneous import  remove_list_from_list, isdict, islist, iscolorimage_dimension, isgrayimage_dimension

def visualize_image(input_image, bgr2rgb=False, save_path=None, vis=False, warning=True, debug=True, closefig=True):
    '''
    visualize an image

    parameters:
        input_image:        a pil or numpy image
        bgr2rgb:            true if the image needs to be converted from bgr to rgb
        save_path:          a path to save. Do not save if it is None
        closefig:           False if you want to add more elements on the image

    outputs:
        fig, ax
    '''
    np_image, _ = safe_image(input_image, warning=warning, debug=debug)
    width, height = np_image.shape[1], np_image.shape[0]
    fig, _ = get_fig_ax_helper(fig=None, ax=None, width=width, height=height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # display image
    if iscolorimage_dimension(np_image):
        if bgr2rgb: image_bgr2rgb(np_image)
        ax.imshow(np_image, interpolation='nearest')
    elif isgrayimage_dimension(np_image):
        np_image = np_image.reshape(np_image.shape[0], np_image.shape[1])
        ax.imshow(np_image, interpolation='nearest', cmap='gray')
    else:
        assert False, 'unknown image type'

    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_image_with_pts(input_image, input_pts, color_index=0, pts_size=20, vis_threshold=-10000, label=False, label_list=None, label_size=20, 
    bgr2rgb=False, save_path=None, vis=False, warning=True, debug=True, closefig=True):
    '''
    visualize an image and plot keypoints on top of it

    parameters:
        input_image:    a pil or numpy image
        input_pts:      2(3) x num_pts numpy array or a dictionary of 2(3) x num_pts array
                        when there are 3 channels in pts, the third one denotes the occlusion/confidence flag
                        occlusion: 0 -> invisible and not annotated, 1 -> visible and annotated, -1 -> visible but not annotated
        color_index:    a scalar or a list of color indexes
        vis_threshold:  the points with confidence above the threshold will be drawn
        label:          determine to add text label for each point
        label_list:     label string for all points

    outputs:
        fig, ax
    '''
    fig, ax = visualize_image(input_image, bgr2rgb=bgr2rgb, vis=False, warning=warning, debug=debug, closefig=False)
    if isdict(input_pts):
        for pts_id, pts_array_tmp in input_pts.items():
            visualize_pts_array(pts_array_tmp, fig=fig, ax=ax, color_index=color_index, pts_size=pts_size, label=label, label_list=label_list, label_size=label_size, 
                plot_occl=True, vis_threshold=vis_threshold, debug=debug, closefig=False)
            color_index += 1
    else: visualize_pts_array(input_pts, fig=fig, ax=ax, color_index=color_index, pts_size=pts_size, label=label, label_list=label_list, label_size=label_size, 
        plot_occl=True, vis_threshold=vis_threshold, debug=debug, closefig=False)

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_image_with_bbox(image, bbox, ax=None, vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize image and plot keypoints on top of it

    parameter:
        image:          a path to an image / an image
        bbox:           N X 4 numpy array, with TLBR format
    '''
    if debug: assert not islist(image), 'this function only support to plot points on one image'
    fig, ax = visualize_image(image, vis=False, debug=debug, closefig=False)
    return visualize_bbox(bbox, fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_image_with_pts_bbox(image, pts_array, window_size, pts_size=20, label=False, label_list=None, color_index=0, vis=False, save_path=None, debug=True, closefig=True):
    '''
    plot a set of points on top of an image with bbox around all points

    parameters
        pts_array:              2 x N
    '''
    # if debug:
        # assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'input points are not correct'

    fig, ax = visualize_image_with_pts(image, pts_array, pts_size=pts_size, label=label, label_list=label_list, color_index=color_index, debug=False, save_path=None, closefig=False)
    bbox = get_center_crop_bbox(pts_array, window_size, window_size, debug=debug)
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
        # assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'input points are not correct'
        if pts_anno is not None: assert pts_array.shape == pts_anno.shape, 'the input points from prediction and annotation have to have the same shape'
        assert islist(valid_index), 'the valid index is not a list'

    num_pts_all = pts_array.shape[1]
    num_pts_succeed = len(valid_index)
    if debug: assert num_pts_succeed <= num_pts_all, 'the number of points should be less than number of points in total'

    color_anno_index = 16                         # cyan
    color_succeed_index = 15                       # yellow
    color_failed_index = 0                        # aqua
    failed_index = remove_list_from_list(range(num_pts_all), valid_index, debug=debug)
    pts_succeed = pts_array[:, valid_index]
    pts_failed = pts_array[:, failed_index]

    # plot successful predictions
    fig, ax = visualize_image_with_pts(image, pts_succeed, pts_size=pts_size, color_index=color_succeed_index, debug=False, closefig=False)
    bbox = get_center_crop_bbox(pts_succeed, window_size, window_size, debug=debug)
    fig, ax = visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_succeed_index, vis=vis, debug=debug, closefig=False)

    # plot failed predictions
    fig, ax = visualize_pts_array(pts_failed, fig=fig, ax=ax, color_index=color_failed_index, pts_size=pts_size, debug=debug, closefig=False)
    bbox = get_center_crop_bbox(pts_failed, window_size, window_size, debug=debug)
    
    if pts_anno is None: return visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_failed_index, vis=vis, save_path=save_path, debug=debug, closefig=closefig)    
    else:
        fig, ax = visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_failed_index, vis=vis, debug=debug, closefig=False)    

        # plot annotations
        fig, ax = visualize_pts_array(pts_anno, fig=fig, ax=ax, color_index=color_anno_index, pts_size=pts_size, debug=debug, closefig=False)
        bbox = get_center_crop_bbox(pts_anno, window_size, window_size, debug=debug)
        return visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_anno_index, vis=vis, save_path=save_path, debug=debug, closefig=closefig)