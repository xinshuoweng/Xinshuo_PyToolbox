% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% test python wrapper in matlab

function test_python_wrapper()
	clc;
	close all;
	clear;

	% init path
	addpath('../');
	py.init_paths.main()

	matrix = rand(2, 4);
	python_np_obj = convert_to_numpy_array(matrix);
	back_matrix = get_numpy_array_from_python(python_np_obj);
	test_matrix = back_matrix ~= matrix;
	assert(sum(test_matrix(:)) == 0, 'test failed!');

	processed = py.python_wrapper.toy_function(python_np_obj);
	back_matrix = get_numpy_array_from_python(processed);
	test_matrix = back_matrix ~= (matrix + 1);
	assert(sum(test_matrix(:)) == 0, 'test failed!');	
end


