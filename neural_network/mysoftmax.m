function output = mysoftmax(net)

% input is the matrix, which contains all linear combination from previous layer
% output is the matrix, by passing the input into the sigmoid function

total = sum(exp(net));
output = exp(net)./total;

end