% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function unnormalizes the label for training

function unnormalized_label = unnormalize_labels(labels, data_range)
	assert(isvector(labels), 'only 1-d labels is supported in this function.');
	assert(numel(data_range) == 2, 'data range is not correct.');

	MAX_LABEL = data_range(2);
	MIN_LABEL = data_range(1);

	unnormalized_label = labels .* (MAX_LABEL - MIN_LABEL);
	unnormalized_label = labels + MIN_LABEL;
end