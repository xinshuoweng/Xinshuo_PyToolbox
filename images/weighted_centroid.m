% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% Analyze noisy 2D images and find peaks using weighted centroids (sub-pixel resolution).
%
% parameters:
%   heatmap:        The 2D data raw image - assumes a Double\Single-precision
%                   floating-point, uint8 or unit16 array. Please note that the code
%                   casts the raw image to uint16 if needed.  If the image dynamic range
%                   is between 0 and 1, I multiplied to fit uint16. This might not be
%                   optimal for generic use, so modify according to your needs.
%   thres:          scalar between 0 and max(raw_image(:)) to remove background
%
% return
%   centroids:      a 2 x N matrix of coordinates of all candidate peaks 
function centroids = weighted_centroid(heatmap, thres, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    % in case of color images
    if ndims(heatmap) > 2 
        heatmap = uint16(rgb2gray(heatmap));
    end

    % for the case the input image is double, casting to uint16 keeps enough dynamic range while speeds up the code.
    if isfloat(heatmap) 
        if max(heatmap(:)) <= 1
            heatmap = uint16(heatmap.*2^16./(max(heatmap(:))));
        else
            heatmap = uint16(heatmap);
        end
    end

    if (nargin < 2)
        thres = (max([min(max(heatmap,[],1))  min(max(heatmap,[],2))]));
    else
        if debug_mode
            assert(thres >= 0 && thres <= 1, 'the threshold should be in the range of [0, 1]\n');
        end
    end

    % analyze image
    centroids = zeros(2, 0);
    if any(heatmap(:))    %for the case of non zero raw image
        % heatmap = medfilt2(heatmap,[3,3]);
        
        % apply threshold
        if isa(heatmap, 'uint8')
            thres = uint8(thres * 255);         % convert floating threshold to uint8
            heatmap = heatmap .* uint8(heatmap > thres);
        else
            thres = uint16(thres * 65535);      % convert floating threshold to uint16
            heatmap = heatmap .* uint16(heatmap > thres);
        end
        
        % find the weighted average center
        if any(heatmap(:))    %for the case of the image is still non zero
            heatmap = single(heatmap);
            binary_mask = logical(heatmap);       % get peak area
            region = regionprops(binary_mask, 'PixelIdxList', 'PixelList');          % get pixel from the area

            % go through all region candidates
            for region_id = 1:length(region)
                pixel_id = region(region_id).PixelIdxList;
                sum_region = sum(heatmap(pixel_id));
                region_x = region(region_id).PixelList(:, 1);
                region_y = region(region_id).PixelList(:, 2);
                centroid_x = sum(region_x .* double(heatmap(pixel_id))) / sum_region;
                centroid_y = sum(region_y .* double(heatmap(pixel_id))) / sum_region;
                centroids(1, region_id) = centroid_x;
                centroids(2, region_id) = centroid_y;
            end 
        end
    end
end
