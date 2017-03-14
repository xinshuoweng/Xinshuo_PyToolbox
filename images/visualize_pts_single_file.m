
function ret = visualize_pts_single_file(image, pts_path)
    assert(ischar(image), 'The input image path is not valid.');
    assert(ischar(pts_path), 'The input point path is not valid.');
    im = imread(image);
    figure;
    imshow(im);
    hold on;
    fid = fopen(pts_path, 'r');
    tline = fgetl(fid);
    while ischar(tline)
        contents = strsplit(tline, ' ');
        plot(str2double(contents{1}), str2double(contents{2}), ...
            'Marker', '*', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
        tline = fgetl(fid);
    end
       
    fclose(fid);
    ret = true;
end
