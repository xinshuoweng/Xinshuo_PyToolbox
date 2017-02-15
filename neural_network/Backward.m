function [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a)
% [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a) computes the gradient
% updates to the deep network parameters and returns them in cell arrays
% 'grad_W' and 'grad_b'. This function takes as input:
%   - 'W' and 'b' the network parameters
%   - 'X' and 'Y' the single input data sample and ground truth output vector,
%     of sizes Nx1 and Cx1 respectively
%   - 'act_a' and 'act_h' the network layer pre and post activations when forward
%     forward propogating the input smaple 'X'

number_hidden = length(act_h);
grad_W = cell(size(W));
grad_b = cell(size(b));

% number_layer = number_hidden + 2;
output = mysoftmax(W{number_hidden+1} * act_h{number_hidden} + b{number_hidden+1});

% compute the gradient of the final output layer
delta = output - Y;          % 9x1
grad_W{number_hidden + 1} = delta * act_h{number_hidden}';   % 7x1 * 1x9
grad_b{number_hidden + 1} = delta;                              % 9x1

% iteratively compute the gradient for all hidden layers
for i = number_hidden:-1:1
    weight_cur = W{i+1}';    % 7x9
    delta_cur = mysigmoidprime(act_h{i}) .* (weight_cur * delta);    % 7x1
    
    if i == 1
        grad_W{i} = delta_cur * X';                                                 
    else
        grad_W{i} = delta_cur * act_h{i-1}';                         % 5x1 * (7x1)'
    end
    
    grad_b{i} = delta_cur;
    delta = delta_cur;
end

end
