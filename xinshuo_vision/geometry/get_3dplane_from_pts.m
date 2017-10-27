% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function takes two lines in and output the point in 2d space
% parameters
%	pts1:	1 x 3			(x, y, z)
%	pts2: 	1 x 3
%	pts3: 	1 x 3
%
% output
%	plane_3d:	1 x 4 		ax + by + cz + d = 0
function plane_3d = get_3dplane_from_pts(pts1, pts2, pts3, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	epsilon = 1e-10;
	if debug_mode
		assert(all(size(pts1) == [1, 3]), 'the size of input points is not correct');
		assert(all(size(pts2) == [1, 3]), 'the size of input points is not correct');
		assert(all(size(pts3) == [1, 3]), 'the size of input points is not correct');
	end

	line1 = pts1 - pts2;
	line2 = pts1 - pts3;

	normal = cross(line1, line2);
	d = pts1(1) * normal(1) + pts1(2) * normal(2) + pts1(3) * normal(3);
	d = -d;

	line1 = line1 ./ line1(1, 3);
	line2 = line2 ./ line2(1, 3);

	if debug_mode
		residual = line1 - line2;
		assert(norm(residual) > epsilon, 'the input 3 points are co-linear');
	end

	plane_3d = [normal, d];
	plane_3d = plane_3d ./ d;				% 1 x 4

	if debug_mode
		assert([pts1, 1] * plane_3d' < epsilon, 'the point is not on the plane');
		assert([pts2, 1] * plane_3d' < epsilon, 'the point is not on the plane');
		assert([pts3, 1] * plane_3d' < epsilon, 'the point is not on the plane');	
	end
end