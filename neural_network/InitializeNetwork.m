function [W, b] = InitializeNetwork(layers)
% InitializeNetwork([INPUT, HIDDEN1, HIDDEN2, ..., OUTPUT]) initializes the weights and biases
% for a fully connected neural network with input data size INPUT, output data
% size OUTPUT, and in between are the number of hidden units in each of the layers.
% It should return the cell arrays 'W' and 'b' which contain the randomly
% initialized weights and biases for this neural network.


% initialization
number_layer = length(layers);
W = cell(1, number_layer-1);
b = cell(1, number_layer-1);

for i = 1:number_layer-1
    % compute N_in and N_out for W at each layer
    W{i} = normrnd(0, 2/(layers(i) + layers(i+1)), [layers(i+1), layers(i)]);
	b{i} = normrnd(0, 2/(layers(i+1) + 1), [layers(i+1), 1]);
end


end
