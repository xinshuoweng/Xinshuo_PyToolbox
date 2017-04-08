% This code implemented a comparison between “k-means” “mean-shift” and
% “normalized-cut” segmentation

% Teste methods are:
% Kmeans segmentation using (color) only
% Kmeans segmentation using (color + spatial)
% Mean Shift segmentation using (color) only
% Mean Shift segmentation using (color + spatial)
% Normalized Cut (inherently uses spatial data)

% an implementation by "Naotoshi Seo" with a little modification is used 
% for “normalized-cut” segmentation, available online at:
% "http://note.sonots.com/SciSoftware/NcutImageSegmentation.html"
% it is sensitive in choosing parameters.
% an implementation by "Bryan Feldman" is used for “mean-shift clustering" 

% Alireza Asvadi
% Department of ECE, SPR Lab
% Babol (Noshirvani) University of Technology
% http://www.a-asvadi.ir
% 2013
%% clear command windows
clc
clear all
close all
%% input
I    = imread('1.jpg');    % Original: also test 2.jpg
%% parameters
% kmeans parameter
K    = 8;                  % Cluster Numbers
% meanshift parameter
bw   = 0.2;                % Mean Shift Bandwidth
% ncut parameters
SI   = 5;                  % Color similarity
SX   = 6;                  % Spatial similarity
r    = 1.5;                % Spatial threshold (less than r pixels apart)
sNcut = 0.21;              % The smallest Ncut value (threshold) to keep partitioning
sArea = 80;                % The smallest size of area (threshold) to be accepted as a segment
%% compare
Ikm          = Km(I,K);                     % Kmeans (color)
Ikm2         = Km2(I,K);                    % Kmeans (color + spatial)
[Ims, Nms]   = Ms(I,bw);                    % Mean Shift (color)
[Ims2, Nms2] = Ms2(I,bw);                   % Mean Shift (color + spatial)
[Inc, Nnc]   = Nc(I,SI,SX,r,sNcut,sArea);   % Normalized Cut
%% show
figure()
subplot(231); imshow(I);    title('Original'); 
subplot(232); imshow(Ikm);  title(['Kmeans',' : ',num2str(K)]);
subplot(233); imshow(Ikm2); title(['Kmeans+Spatial',' : ',num2str(K)]); 
subplot(234); imshow(Ims);  title(['MeanShift',' : ',num2str(Nms)]);
subplot(235); imshow(Ims2); title(['MeanShift+Spatial',' : ',num2str(Nms2)]);
subplot(236); imshow(Inc);  title(['NormalizedCut',' : ',num2str(Nnc)]); 

