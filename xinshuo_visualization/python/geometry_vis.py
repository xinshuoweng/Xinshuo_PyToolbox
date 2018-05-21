# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, matplotlib.pyplot as plt, colorsys
import matplotlib.collections as plycollections
from matplotlib.patches import Ellipse
# from scipy.stats import norm, chi2

from private import save_vis_close_helper, get_fig_ax_helper
from xinshuo_math.python.private import safe_2dptsarray

from xinshuo_math import pts_euclidean, bbox_TLBR2TLWH, bboxcheck_TLBR
from xinshuo_miscellaneous import islogical, islist, isstring, is2dptsarray_confidence, is2dptsarray_occlusion, is2dptsarray, isdict, list_reorder, list2tuple

color_set = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w', 'lime', 'cyan', 'aqua']
color_set_big = ['aqua', 'azure', 'red', 'black', 'blue', 'brown', 'cyan', 'darkblue', 'fuchsia', 'gold', 'green', 'grey', 'indigo', 'magenta', 'lime', 'yellow', 'white', 'tomato', 'salmon']
marker_set = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
hatch_set = [None, 'o', '/', '\\', '|', '-', '+', '*', 'x', 'O', '.']
linestyle_set = ['-', '--', '-.', ':', None, ' ', 'solid', 'dashed']
dpi = 80

def visualize_pts_array(input_pts, covariance=False, color_index=0, pts_size=20, label=False, label_list=None, label_size=2, plot_occl=True, vis_threshold=-10000, 
    xlim=None, ylim=None, fig=None, ax=None, save_path=None, vis=False, warning=True, debug=True, closefig=True):
    '''
    plot keypoints with covariance ellipse

    parameters:
        pts_array:      2(3) x num_pts numpy array, the third channel could be confidence or occlusion
    '''
    # obtain the points
    try: pts_array = safe_2dptsarray(input_pts, homogeneous=True, warning=warning, debug=debug)
    except AssertionError: pts_array = safe_2dptsarray(input_pts, homogeneous=False, warning=warning, debug=debug)
    if debug: assert is2dptsarray(input_pts) or is2dptsarray_occlusion(input_pts) or is2dptsarray_confidence(input_pts), 'input points are not correct'
    num_pts = pts_array.shape[1]

    # obtain a label list if required but not provided
    if debug: assert islogical(label), 'label flag is not correct'
    if label and (label_list is None): label_list = [str(i+1) for i in xrange(num_pts)]
    if label and debug: assert islist(label_list) and all(isstring(label_tmp) for label_tmp in label_list), 'labels are not correct'

    # obtain the color index
    if islist(color_index):
        if debug: assert not (plot_occl or covariance) , 'the occlusion or covariance are not compatible with plotting different colors during scattering'
        color_tmp = [color_set_big[index_tmp] for index_tmp in color_index]
    else: color_tmp = color_set_big[color_index % len(color_set_big)]
    
    fig, ax = get_fig_ax_helper(fig=fig, ax=ax)
    std, conf = None, 0.95
    if is2dptsarray(pts_array):             # only 2d points without third rows
        if debug and islist(color_tmp): assert len(color_tmp) == num_pts, 'number of points to plot is not equal to number of colors provided'
        ax.scatter(pts_array[0, :], pts_array[1, :], color=color_tmp, s=pts_size)
        pts_visible_index = range(pts_array.shape[1])
        pts_ignore_index = []
        pts_invisible_index = []
    else:
        num_float_elements = np.where(np.logical_and(pts_array[2, :] > 0, pts_array[2, :] < 1))[0].tolist()
        if len(num_float_elements) > 0: type_3row = 'conf'
        else: type_3row = 'occu'

        if type_3row == 'occu':
            pts_visible_index   = np.where(pts_array[2, :] == 1)[0].tolist()              # plot visible points in red color
            pts_ignore_index    = np.where(pts_array[2, :] == -1)[0].tolist()             # do not plot points with annotation, usually visible, but not annotated
            pts_invisible_index = np.where(pts_array[2, :] == 0)[0].tolist()              # plot invisible points in blue color
        else:
            pts_visible_index   = np.where(pts_array[2, :] > vis_threshold)[0].tolist()
            pts_ignore_index    = np.where(pts_array[2, :] <= vis_threshold)[0].tolist()
            pts_invisible_index = []

        if debug and islist(color_tmp): assert len(color_tmp) == len(pts_visible_index), 'number of points to plot is not equal to number of colors provided'
        ax.scatter(pts_array[0, pts_visible_index], pts_array[1, pts_visible_index], color=color_tmp, s=pts_size)
        if plot_occl: ax.scatter(pts_array[0, pts_invisible_index], pts_array[1, pts_invisible_index], color=color_set_big[(color_index+1) % len(color_set_big)], s=pts_size)
        if covariance: visualize_pts_covariance(pts_array[0:2, :], std=std, conf=conf, fig=fig, ax=ax, debug=debug, color=color_tmp)

    if label:
        for pts_index in xrange(num_pts):
            label_tmp = label_list[pts_index]
            if pts_index in pts_ignore_index: continue
            else:
                # note that the annotation is based on the coordinate instead of the order of plotting the points, so the orider in pts_index does not matter
                if islist(color_index): plt.annotate(label_tmp, xy=(pts_array[0, pts_index], pts_array[1, pts_index]), xytext=(-1, 1), color=color_set_big[(color_index[pts_index]+5) % len(color_set_big)], textcoords='offset points', ha='right', va='bottom', fontsize=label_size)
                else: plt.annotate(label_tmp, xy=(pts_array[0, pts_index], pts_array[1, pts_index]), xytext=(-1, 1), color=color_set_big[(color_index+5) % len(color_set_big)], textcoords='offset points', ha='right', va='bottom', fontsize=label_size)
    
    # set axis
    if xlim is not None:
        if debug: assert islist(xlim) and len(xlim) == 2, 'the x lim is not correct'
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:    
        if debug: assert islist(ylim) and len(ylim) == 2, 'the y lim is not correct'
        plt.ylim(ylim[0], ylim[1])

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig, transparent=False)

def visualize_lines(lines_array, color_index=0, line_width=3, fig=None, ax=None, vis=True, save_path=None, debug=True, closefig=True):
    '''
    plot lines 

    parameters:
        lines_array:            4 x num_lines, each column denotes (x1, y1, x2, y2)
    '''

    if debug: assert islinesarray(lines_array), 'input array of lines are not correct'
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
        assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array) or is2dptsarray_confidence(pts_array), 'input points are not correct'
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
    width, height = 1024, 1024
    fig, _ = get_fig_ax_helper(fig=fig, ax=ax, width=width, height=height)
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

def visualize_bbox(bbox, fig=None, ax=None, linewidth=0.5, color_index=20, vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize a set of bounding box

    parameters:
        bbox:       N x 4
    '''
    if debug: assert bboxcheck_TLBR(bbox), 'input bounding boxes are not correct'
    edge_color = color_set_big[color_index % len(color_set_big)]

    # plot bounding box
    bbox = bbox_TLBR2TLWH(bbox, debug=debug)              # convert TLBR format to TLWH format
    for bbox_index in range(bbox.shape[0]):
        bbox_tmp = bbox[bbox_index, :]     
        ax.add_patch(plt.Rectangle((bbox_tmp[0], bbox_tmp[1]), bbox_tmp[2], bbox_tmp[3], fill=False, edgecolor=edge_color, linewidth=linewidth))

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
        if conf is not None: assert isscalar(conf) and conf >= 0 and conf <= 1, 'the confidence is not in a good range'
        if std is not None: assert ispositiveinteger(std), 'the number of standard deviation should be a positive integer'

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