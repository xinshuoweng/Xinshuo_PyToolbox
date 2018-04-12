% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function takes 3d point and number of requested plane, output the 3d plane in the format of ax + by + cz + d = 0
% note that the criteria for stopping the RANSAC is only the number of inliers now, do not consider the standard deviation for each plane
% be careful about the err_threshold!!!! choose a good value is key to succeed as the algorithm might detect the same plane for multiple times if the 
% threshold is too strict
%
% input
%			pts_3d					num_pts x 3				(x, y, z)
%
% output
%			plane_3d 				num_plane x 4
%			pts_index_plane 		num_pts x 1, 			the index of plane belonging to for every pts, 0 indicate bad pts
%			corresponding_pts 		num_plane x 1 cell, 	save the ptss for every VP
function [plane_3d, pts_index_plane, corresponding_pts] = get_dominant_3dplane_RANSAC(pts_3d, num_plane, debug_mode, vis_mode, save_path, max_iter, err_threshold)
	color_set = ['r', 'g', 'b', 'k', 'y', 'm', 'c', 'w'];
	if nargin < 7
		err_threshold = 0.1;
	end

	if nargin < 6
		max_iter = 50000;
	end

	if nargin < 5
		save_path = '';
	end

	if nargin < 4
		vis_mode = false;
	end

	if nargin < 3 
		debug_mode = true;
	end

	epsilon = 1e-5;

	if debug_mode
		assert(is3dPtsArray(pts_3d'), 'the input 3d points are not correct');
		assert(isInteger(num_plane), 'the input number of plane should be an integer');
	end

	num_pts = size(pts_3d, 1);
	pts_index_plane = zeros(num_pts, 1);
	plane_3d = zeros(num_plane, 4);						% num_plane x 3
	num_inlier_plane = zeros(num_plane, 1);				% number of inlier per plane
	best_num_inlier = 0;
	valid = false;
	corresponding_pts = cell(num_plane, 1);
	corresponding_pts_index = cell(num_plane, 1);
	for iter_index = 1:max_iter
		while 1 				% sometimes the sampled pts are parallel 
			try
				sampled_index = randsample(num_pts, num_plane*3);
				pts_sampled = pts_3d(sampled_index, :);
				
				% compute the temperorary plane_3d
				for plane_index = 1:num_plane
					pts1_tmp = pts_sampled((plane_index-1)*3 + 1, :);
					pts2_tmp = pts_sampled((plane_index-1)*3 + 2, :);
					pts3_tmp = pts_sampled(plane_index*3, :);
					plane_3d(plane_index, :) = get_3dplane_from_pts(pts1_tmp, pts2_tmp, pts3_tmp, debug_mode);
				end
				break;
			catch 
				% pause;
				fprintf('current sample is not good, resample......\n');
				continue;
			end
		end

		dist_matrix = get_distance_from_pts_3dplane(pts_3d', plane_3d, debug_mode);			% num_plane x num_pts
		dist_matrix = dist_matrix';															% num_pts x num_plane
		% dist_matrix
		% pause;
		[min_dist, min_dist_index] = min(dist_matrix, [], 2);
		inlier_index = find(min_dist < err_threshold);
		num_inliers = length(inlier_index);
		pts_index_plane(inlier_index, :) = min_dist_index(inlier_index);					% assign the correst plane to the inlier

		% get information for every VP
		for plane_index = 1:num_plane
			inlier_index_pts_tmp = inlier_index(find(min_dist_index(inlier_index) == plane_index));
			num_inlier_plane(plane_index) = length(inlier_index_pts_tmp);
			corresponding_pts{plane_index, 1} = pts_3d(inlier_index_pts_tmp, :);
			corresponding_pts_index{plane_index, 1} = inlier_index_pts_tmp;
		end
		assert(sum(num_inlier_plane) == num_inliers, 'the number of inlier for every plane is wrong');

		% criteria to improve the VP
		if num_inliers > best_num_inlier
			best_num_inlier = num_inliers;
			best_plane = plane_3d;
			best_pts_index_plane = pts_index_plane;
			best_corresponding_pts = corresponding_pts;
			valid = true;
			% fprintf('best number of inlier')
			fprintf('iter: %d, best number of inliers is %d\n', iter_index, best_num_inlier);
		end

		
		if num_inliers > 0.95 * num_pts
			break;
		end
	end
	assert(valid, 'no good plane found after %d iterations\n', max_iter);
	
	plane_3d = best_plane(:, 1:2);
	corresponding_pts = best_corresponding_pts;
	pts_index_plane = best_pts_index_plane;

	% visualization
	if vis_mode
		fig = figure; 
		for plane_index = 1:num_plane
			color_index = plane_index;
			color_tmp = color_set(color_index);
			pts_tmp = corresponding_pts{plane_index};
			scatter3(pts_tmp(:, 1), pts_tmp(:, 2), pts_tmp(:, 3), 'MarkerEdgeColor', color_tmp, 'MarkerFaceColor', color_tmp);
			hold on; 
		end
		hold off;
		pause;
		if ~isempty(save_path)
			print(save_path, '-depsc');
		end
	end
end