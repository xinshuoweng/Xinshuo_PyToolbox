function [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate)
% [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate) computes and returns the
% new network parameters 'W' and 'b' with respect to the old parameters, the
% gradient updates 'grad_W' and 'grad_b', and the learning rate.

%%%% minus or addition should be waiting for checking!!!!!!!!!!!!!!!!!

number_hidden = length(W)-1;
for i = 1:number_hidden+1
    W{i} = W{i} - learning_rate.*grad_W{i};
    b{i} = b{i} - learning_rate.*grad_b{i};
end

end

