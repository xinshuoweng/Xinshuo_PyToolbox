% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function gets a numpy array object in matlab 
% and convert it to a matrix with same shape and data

function [data] = get_numpy_array_from_python(np_array_obj)
	assert(isa(np_array_obj, 'py.numpy.ndarray'), 'input should be a numpy array from python');
	dtype = char(tostring(py.array.array('c',py.str(np_array_obj.dtype.name))));
	pydata = double(py.array.array('d', py.numpy.nditer(np_array_obj)));
	shape = double(py.array.array('I',py.tuple(np_array_obj.shape)));
	assert(length(shape) <= 2, 'only 2-d numpy array is supported now.');

	data = reshape(pydata, fliplr(shape))';
	test_dimen = (size(data) ~= shape);
	assert(sum(test_dimen(:)) == 0, 'shape is not correct');
end