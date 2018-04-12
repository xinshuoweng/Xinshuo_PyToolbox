% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function takes an image in and output the dominate vanishing points
% note that the criteria for stopping the RANSAC is only the number of inliers now, do not consider the standard deviation for each VP
% be careful about the err_threshold!!!! choose a good value is key to succeed as the algorithm might detect the same VP for multiple times if the 
% threshold is too strict
%
% output
%			VPs 					num_vp x 2
%			lines_seg 				num_lines x 6, [x1 x2 y1 y2 theta r]
%			line_index 				num_lines x 1, 		the index of VP belonging to for every line, 0 indicate bad line
%			corresponding_lines 	num_vp x 1 cell, 	save the lines for every VP
function [VPs, lines_seg, line_index, corresponding_lines] = get_vanishing_points_RANSAC(img, num_vp, debug_mode, vis_mode, max_iter, err_threshold)
	color_set = ['r', 'g', 'b', 'k', 'y', 'm', 'c', 'w'];
	if nargin < 6
		err_threshold = 50;
	end

	if nargin < 5
		max_iter = 50000;
	end

	if nargin < 4
		vis_mode = false;
	end

	if nargin < 3 
		debug_mode = true;
	end

	if nargin < 2
		num_vp = 3;
	end
	epsilon = 1e-5;

	% extract the line segments from the cluster_indeximage
	im_gray = rgb2gray(img);
	im_height = size(im_gray, 1);
	im_width = size(im_gray, 2);
	length_diagonal = sqrt(im_height ^ 2 + im_width ^ 2);
	lines_seg = APPgetLargeConnectedEdges(im_gray, 0.025 * length_diagonal);			% num_lines x 7
	% lines = APPgetLargeConnectedEdges(im_gray, 30);			% num_lines x 7
	num_lines = size(lines_seg, 1);
	fprintf('%d lines found in the image\n', num_lines);
	lines_2d = get_2dline_from_segment(lines_seg, debug_mode);

	line_index = zeros(num_lines, 1);
	VPs = zeros(num_vp, 3);						% num_pts x 3
	num_inlier_VP = zeros(num_vp, 1);			% number of inlier per VP
	best_num_inlier = 0;
	valid = false;
	corresponding_lines = cell(num_vp, 1);
	corresponding_lines_index = cell(num_vp, 1);
	for iter_index = 1:max_iter
		while 1 				% sometimes the sampled line are parallel 
			try
				sampled_index = randsample(num_lines, num_vp*2);
				lines_sampled = lines_2d(sampled_index, :);
				
				% compute the temperorary VPs
				for cluster_index = 1:num_vp
					line1_tmp = lines_sampled((cluster_index-1)*2 + 1, :);
					line2_tmp = lines_sampled(cluster_index*2, :);
					VPs(cluster_index, :) = get_pts_from_2dlines(line1_tmp, line2_tmp, debug_mode);
				end
				break;
			catch 
				fprintf('current sample is not good, resample......\n');
				continue;
			end
		end

		dist_matrix = get_distance_from_pts_2dline(VPs', lines_2d, debug_mode);			% num_lines x num_vp
		[min_dist, min_dist_index] = min(dist_matrix, [], 2);
		inlier_index = find(min_dist < err_threshold);
		num_inliers = length(inlier_index);
		line_index(inlier_index, :) = min_dist_index(inlier_index);					% assign the correst VP to the inlier

		% get information for every VP
		for cluster_index = 1:num_vp
			inlier_index_vp_tmp = inlier_index(find(min_dist_index(inlier_index) == cluster_index));
			num_inlier_VP(cluster_index) = length(inlier_index_vp_tmp);
			corresponding_lines{cluster_index, 1} = lines_seg(inlier_index_vp_tmp, :);
			corresponding_lines_index{cluster_index, 1} = inlier_index_vp_tmp;
		end
		assert(sum(num_inlier_VP) == num_inliers, 'the number of inlier for every VP is wrong');

		% criteria to improve the VP
		fprintf('iter: %d, best number of inliers is %d\n', iter_index, best_num_inlier);
		if num_inliers > best_num_inlier
			best_num_inlier = num_inliers;
			best_vp = VPs;
			best_lines_index = line_index;
			best_corresponding_lines = corresponding_lines;
			valid = true;
		end

		if num_inliers > 0.7 * num_lines
			break;
		end
	end
	assert(valid, 'no good VP found after %d iterations\n', max_iter);
	
	VPs = best_vp(:, 1:2);
	corresponding_lines = best_corresponding_lines;
	line_index = best_lines_index;

	% visualization
	if vis_mode
		save_path = '';
		label = false;
		label_str = '';
		vis_radius = 3;
		vis_resize_factor = 1;
		closefig = false;
		for cluster_index = 1:num_vp
			color_index = cluster_index;
			img_with_pts = visualize_image_with_pts(img, VPs(cluster_index, :)', vis_mode, debug_mode, save_path, label, label_str, vis_radius, vis_resize_factor, closefig, color_index);
			hold on; plot(corresponding_lines{cluster_index}(:, [1 2])', corresponding_lines{cluster_index}(:, [3 4])', color_set(color_index));
			hold off;
		end
	end
end