function [accuracy, loss] = ComputeAccuracyAndLoss(W, b, data, labels)
% [accuracy, loss] = ComputeAccuracyAndLoss(W, b, X, Y) computes the networks
% classification accuracy and cross entropy loss with respect to the data samples
% and ground truth labels provided in 'data' and labels'. The function should return
% the overall accuracy and the average cross-entropy loss.

% accuracy is the classification rate

number_examples = size(data, 1);
outputs = Classify(W, b, data);		% D x C
temp1 = outputs .* labels;
temp2 = sum(temp1, 2);
loss_total = -log(temp2);
loss = sum(loss_total)/number_examples;
[~, ex_id] = max(outputs, [], 2);
[~, gt_id] = max(labels, [], 2);
accuracy = (sum(ex_id - gt_id == 0))/size(labels, 1);

end
