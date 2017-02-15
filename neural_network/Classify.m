function [outputs] = Classify(W, b, data)
% [predictions] = Classify(W, b, data) should accept the network parameters 'W'
% and 'b' as well as an DxN matrix of data sample, where D is the number of
% data samples, and N is the dimensionality of the input data. This function
% should return a vector of size DxC of network softmax output probabilities.

number_samples = size(data, 1);
outputs = zeros(number_samples, size(W{length(W)}, 1));
for i = 1:number_samples
    data_temp = data(i, :)';
	outputs(i, :) = Forward(W, b, data_temp)';
end

end
