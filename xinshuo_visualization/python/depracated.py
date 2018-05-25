
# def visualize_image_with_pts_bbox_tracking(image, pts_array, valid_index, window_size, pts_anno=None, pts_size=20, vis=False, save_path=None, debug=True, closefig=True):
#     '''
#     plot a set of points from tracking results on top of an image with bbox, and plot the annotation meanwhile with another color
#     the tracking results also include the successful or failed tracking, we differentiate them in different color

#     parameters:
#         pts_array:              2 x N
#         pts_anno:               2 x N
#         valid_index:            a list of m elements who succeeds, m >= 0 && m <= N
#     '''
#     if debug:
#         # assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'input points are not correct'
#         if pts_anno is not None: assert pts_array.shape == pts_anno.shape, 'the input points from prediction and annotation have to have the same shape'
#         assert islist(valid_index), 'the valid index is not a list'

#     num_pts_all = pts_array.shape[1]
#     num_pts_succeed = len(valid_index)
#     if debug: assert num_pts_succeed <= num_pts_all, 'the number of points should be less than number of points in total'

#     color_anno_index = 16                         # cyan
#     color_succeed_index = 15                       # yellow
#     color_failed_index = 0                        # aqua
#     failed_index = remove_list_from_list(range(num_pts_all), valid_index, debug=debug)
#     pts_succeed = pts_array[:, valid_index]
#     pts_failed = pts_array[:, failed_index]

#     # plot successful predictions
#     fig, ax = visualize_image_with_pts(image, pts_succeed, pts_size=pts_size, color_index=color_succeed_index, debug=False, closefig=False)
#     bbox = get_center_crop_bbox(pts_succeed, window_size, window_size, debug=debug)
#     fig, ax = visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_succeed_index, vis=vis, debug=debug, closefig=False)

#     # plot failed predictions
#     fig, ax = visualize_pts_array(pts_failed, fig=fig, ax=ax, color_index=color_failed_index, pts_size=pts_size, debug=debug, closefig=False)
#     bbox = get_center_crop_bbox(pts_failed, window_size, window_size, debug=debug)
    
#     if pts_anno is None: return visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_failed_index, vis=vis, save_path=save_path, debug=debug, closefig=closefig)    
#     else:
#         fig, ax = visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_failed_index, vis=vis, debug=debug, closefig=False)    

#         # plot annotations
#         fig, ax = visualize_pts_array(pts_anno, fig=fig, ax=ax, color_index=color_anno_index, pts_size=pts_size, debug=debug, closefig=False)
#         bbox = get_center_crop_bbox(pts_anno, window_size, window_size, debug=debug)
#         return visualize_bbox(bbox, fig=fig, ax=ax, color_index=color_anno_index, vis=vis, save_path=save_path, debug=debug, closefig=closefig)