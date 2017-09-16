% Author: Xinshuo Weng
% Email: xinshuo.weng@gmail.com

% add a python module to the system path
function dummy = import_pymodule(module_path)
	assert(isFolder(module_path), 'path of module is not correct');
	P = py.sys.path;
	if count(P, module_path) == 0
	    insert(P, int32(0), module_path);
	end
end