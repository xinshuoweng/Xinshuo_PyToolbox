% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function solves an equation set with two variables
% 	[a11	a12] * [x] = [m]
%	[a21	a22]   [y]	  [n]
%
%	inputs
%			params:		2 x 2 matrix
%			right_size:	2 x 1 matrix
function [x, y] = solve_equation_set(params, right_side, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(all(size(params) == [2, 2]), 'the parameter dimension is not correct');
		assert(all(size(right_side) == [2, 1]), 'the right side has wrong dimension');
	end

	denominator_x = params(2, 2) * params(1, 1) - params(1, 2) * params(2, 1);
	denominator_y = params(2, 1) * params(1, 2) - params(1, 1) * params(2, 2);
	if debug_mode
		assert(abs(denominator_x) > 1e-4, 'the denominator should not be zero'); 
		assert(abs(denominator_y) > 1e-4, 'the denominator should not be zero'); 
	end

	x = (right_side(1) * params(2, 2) - right_side(2) * params(1, 2)) / denominator_x;
	y = (right_side(1) * params(2, 1) - right_side(2) * params(1, 1)) / denominator_y;


	if debug_mode
		solution = [x; y];
		residual = norm(params * solution - right_side);
		assert(residual < 1e-4, 'the solution is not correct');
	end
end