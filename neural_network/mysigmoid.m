function output = mysigmoid(net)

% input is the matrix, which contains all linear combination from previous layer
% output is the matrix, by passing the input into the sigmoid function

output = 1./(1 + exp(-net));

end