% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% get the sparsity of an input array
function [sparsity_ratio, index_0] = compute_sparsity(ndarray)
	vectorized = ndarray(:);
	index_0 = find(vectorized == 0);
	num_elements = length(vectorized);
	sparsity_ratio = length(index_0) / num_elements;
end