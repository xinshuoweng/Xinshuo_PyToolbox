% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% input is the matrix, which contains all linear combination from previous layer
% output is the matrix, by passing the input into the sigmoid function
function output = mysoftmax(net)
	total = sum(exp(net));
	output = exp(net)./total;
end