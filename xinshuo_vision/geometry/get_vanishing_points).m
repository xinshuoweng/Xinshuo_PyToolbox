% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% TODO: check conf, check theta, more carefully compute angular, explore houghtransform

% this function takes an image in and output the dominate vanishing points
function VPs = get_vanishing_points(img, num_vp, max_iter, debug_mode, vis_mode, error_threshold)
	color_set = ['r', 'g', 'b', 'k', 'y', 'm', 'c', 'w'];
	if nargin < 6
		error_threshold = 10;
	end

	if nargin < 5
		vis_mode = false;
	end

	if nargin < 4 
		debug_mode = true;
	end

	if nargin < 3
		max_iter = 10000;
	end

	if nargin < 2
		num_vp = 3;
	end

	% extract the line segments from the image
	im_gray = rgb2gray(img);
	im_height = size(im_gray, 1);
	im_width = size(im_gray, 2);
	length_diagonal = sqrt(im_height ^ 2 + im_width ^ 2);
	% lines = APPgetLargeConnectedEdges(im_gray, 0.025 * length_diagonal);			% num_lines x 7
	lines = APPgetLargeConnectedEdges(im_gray, 30);			% num_lines x 7

	max_iter = 1000;
	num_lines = size(lines, 1);
	sorted_lines = sortrows(lines, 7);
	
	% collect the psudo-centroid
	VPs = zeros(num_vp, 3);
	VPs_old = zeros(num_vp, 3);
	lines_selected = sorted_lines(end-2*num_vp:end, :);								% random select first seed based on quality of lines
	for cluster_index = 1:num_vp
		line1_segment = lines_selected((cluster_index-1)*2 + 1, :);
		line2_segment = lines_selected((cluster_index-1)*2 + 2, :);

		line1_2d = get_2dline_from_segment(line1_segment, debug_mode);
		line2_2d = get_2dline_from_segment(line2_segment, debug_mode);
		pts_infinity = get_pts_from_2dlines(line1_2d, line2_2d, debug_mode);
		VPs(cluster_index, :) = pts_infinity;
	end

	% compute the vanishing point inside each cluster
	for iter_index = 1:max_iter
		fprintf('iter: %d\n', iter_index);

		% assignment step, assign every line to the right cluster
		corresponding_lines = cell(num_vp, 1);
		for line_index = 1:num_lines
			best_distance = realmax;
			line_segment_tmp = lines(line_index, :);
			line_tmp = get_2dline_from_segment(line_segment_tmp, debug_mode);
			right_cluster_index = 0;

			for cluster_index = 1:num_vp
				centroid_tmp = VPs(cluster_index, :);
				distance_tmp = get_distance_from_pts_2dline(centroid_tmp, line_tmp, debug_mode);
				if distance_tmp < best_distance
					best_distance = distance_tmp;
					right_cluster_index = cluster_index;
				end
			end

			assert(right_cluster_index > 0 && right_cluster_index <= num_vp, 'the assignment is wrong');
			corresponding_lines{right_cluster_index} = [corresponding_lines{right_cluster_index}; line_segment_tmp];
		end

		% update step
		for cluster_index = 1:num_vp
			% find the first seed 
			lines_cluster = corresponding_lines{cluster_index};
			num_lines_tmp = size(lines_cluster, 1);

			angular_value = lines_cluster(:, 5);
			[closest_mean, line_seed1_index, mean_value] = find_closest_mean_from_array(angular_value, debug_mode);

			line_seed1 = lines_cluster(line_seed1_index, :);
			line_seed1_proj = get_2dline_from_segment(line_seed1, debug_mode);

			% fine the second seed
			intersection_points = zeros(num_lines_tmp, 3);
			intersection_points_proj = zeros(num_lines_tmp, 1);
			for line_index = 1:num_lines_tmp
				if line_index == line_seed1_index			% skip itself
					continue;
				end

				line1_segment = lines_cluster(line_seed1_index, :);
				line2_segment = lines_cluster(line_index, :);
				
				line1_2d = get_2dline_from_segment(line1_segment, debug_mode);
				line2_2d = get_2dline_from_segment(line2_segment, debug_mode);

				pts_intersection = get_pts_from_2dlines(line1_2d, line2_2d, debug_mode);
				intersection_points(line_index, :) = pts_intersection;

				if line_seed1_proj(2) > 1e-5
					intersection_points_proj(line_index, 1) = pts_intersection(1);			% project to x axis
				else
					intersection_points_proj(line_index, 1) = pts_intersection(2);			% project to y axis
				end
			end
			assert(all(intersection_points(line_seed1_index, :) == [0, 0, 0]), 'the intersection_points is wrong');

			% skip the self
			intersection_points(line_seed1_index, :) = [];
			intersection_points_proj(line_seed1_index) = [];
			
			[closest_mean, line_seed2_index, mean_value] = find_closest_mean_from_array(intersection_points_proj, debug_mode);
			if line_seed2_index >= line_seed1_index
				line_seed2_index = line_seed2_index + 1;
			end

			assert(line_seed2_index ~= line_seed1_index, 'the seed is not correct');

			% update the centroid with the new seeds
			line1_segment = lines_cluster(line_seed1_index, :);
			line2_segment = lines_cluster(line_seed2_index, :);

			line1_2d = get_2dline_from_segment(line1_segment, debug_mode);
			line2_2d = get_2dline_from_segment(line2_segment, debug_mode);
			pts_infinity = get_pts_from_2dlines(line1_2d, line2_2d, debug_mode);
			VPs(cluster_index, :) = pts_infinity;		
		end

		VPs_diff = VPs - VPs_old;
		if norm(VPs_diff) < 1e-5
			break;
		end

		VPs_old = VPs;
		% pause;
% 
	% for cluster_index = 1:num_vp
		% lines_tmp = lines(index_line == cluster_index, :);			% num_lines x 6

	% 	% RANSAC to compute the vanishing point
	% 	num_lines = size(lines_tmp, 1);
	% 	biggest_num_valid = 0;
	% 	for iter_index = 1:num_iter
	% 		line1_index = randi(num_lines);
	% 		line2_index = randi(num_lines-1);
	% 		if line2_index >= line1_index
	% 			line2_index = line2_index + 1;
	% 		end

	% 		pts1 = lines_tmp(line1_index, 1:2);
	% 		pts2 = lines_tmp(line1_index, 3:4);
	% 		pts3 = lines_tmp(line2_index, 1:2);
	% 		pts4 = lines_tmp(line2_index, 3:4);

	% 		line1_2d = get_2dline_from_pts(pts1, pts2, debug_mode);
	% 		line2_2d = get_2dline_from_pts(pts3, pts4, debug_mode);

	% 		pts_infinity = get_pts_from_2dlines(line1_2d, line2_2d, debug_mode);

	% 		% compute the distance for all line within the cluster
	% 		num_valid = 0;
	% 		lines_valid = zeros(num_lines, 6);
	% 		for line_index = 1:num_lines
	% 			pts_test1 = lines_tmp(line_index, 1:2);
	% 			pts_test2 = lines_tmp(line_index, 3:4);
	% 			line_test = get_2dline_from_pts(pts_test1, pts_test2, debug_mode);

	% 			distance = get_distance_from_pts_2dline(pts_infinity, line_test, debug_mode);
	% 			if distance < error_threshold
	% 				num_valid = num_valid + 1;
	% 				lines_valid(num_valid, :) = lines_tmp(line_index, :);
	% 			end
	% 		end

	% 		if num_valid > biggest_num_valid
	% 			biggest_num_valid = num_valid;
	% 			best_vp = pts_infinity;
	% 			best_lines = lines_valid(1:num_valid, :);
	% 		end
	% 	end

	% 	best_vp = best_vp / best_vp(3);
	% 	VPs(cluster_index, :) = best_vp(1:2);
	% 	size(best_lines)
	% 	corresponding_lines{cluster_index} = best_lines;
	% end
	end

	VPs = VPs(:, 1:2);

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
			size(corresponding_lines{cluster_index})
		end
	end
end