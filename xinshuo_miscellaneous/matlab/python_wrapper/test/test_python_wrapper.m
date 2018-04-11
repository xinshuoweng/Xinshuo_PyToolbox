% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% test python wrapper in matlab
function test_python_wrapper()
	% init path
	addpath('../');
	
	% if ~exist('init_paths.py', 'file')
	% 	fprintf('No init_paths.py found in parent folder. PLease create one!!');
	% end
	% py.init_paths.main();

	matrix = rand(2, 4);
	python_np_obj = convert_to_numpy_array(matrix);
	back_matrix = get_numpy_array_from_python(python_np_obj);
	test_matrix = back_matrix ~= matrix;
	assert(sum(test_matrix(:)) == 0, 'test failed!');

	processed = py.xinshuo_matlab.toy_function(python_np_obj);
	back_matrix = get_numpy_array_from_python(processed);
	test_matrix = back_matrix ~= (matrix + 1);
	assert(sum(test_matrix(:)) == 0, 'test failed!');	
end