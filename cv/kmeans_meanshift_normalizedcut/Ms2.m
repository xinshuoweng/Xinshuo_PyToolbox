function [Ims Kms clustMembsCell] = Ms2(I,bandwidth)

%% color + spatial (option: bandwidth)
I = im2double(I);
[x,y] = meshgrid(1:size(I,2),1:size(I,1)); L = [y(:)/max(y(:)),x(:)/max(x(:))]; % Spatial Features
C = reshape(I,size(I,1)*size(I,2),3); X = [C,L];                                % Color & Spatial Features
%% MeanShift Segmentation
[clustCent,point2cluster,clustMembsCell] = MeanShiftCluster(X',bandwidth);      % MeanShiftCluster


for i = 1:length(clustMembsCell)                                                % Replace Image Colors With Cluster Centers
X(clustMembsCell{i},:) = repmat(clustCent(:,i)',size(clustMembsCell{i},2),1); 
end
Ims = reshape(X(:,1:3),size(I,1),size(I,2),3);                                  % Segmented Image
Kms = length(clustMembsCell);

end