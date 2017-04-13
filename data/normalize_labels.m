% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function normalizes the label for training

function normalized_label = normalize_labels(labels, data_range)
	assert(isvector(labels), 'only 1-d labels is supported in this function.');
	assert(numel(data_range) == 2, 'data range is not correct.');

	if ~exist('data_range', 'var')
		MAX_LABEL = max(labels);
		MIN_LABEL = min(labels);
	else
		MAX_LABEL = data_range(2);
		MIN_LABEL = data_range(1);
	end

	normalized_labels = labels + MIN_LABEL;
	normalized_labels = normalized_labels ./ (MAX_LABEL - MIN_LABEL);
end
