% Author: Xinshuo Weng
% Email: xinshuo.weng@gmail.com

% visualize the blob of caffe net
function caffe_blob_visualization(net, blob_name)
	assert(ischar(blob_name), 'blob name is not correct');

	blob = net.blobs(blob_name).get_data();
	assert(size(blob, 3) == 3, 'this blob is not 3-channel blob and cannot be visualized naively.');
	blob = permute(blob, [2, 1, 3]);   % from W x H x C to H x W x C
	fprintf('%s blob shape: ', blob_name);
	disp(size(blob));

	figure;
	imshow(blob);
end