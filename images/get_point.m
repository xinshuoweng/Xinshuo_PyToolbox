% this function

function ret = get_point(image, number_point, save_path)
    im = imread(image);
    imshow(im);
    pts = ginput(number_point);
    close all;
    pts(:, 3) = 0.5;
    fid = fopen(save_path, 'w');
    for i = 1:number_point
        fprintf(fid, '%d %d %d\n', pts(i, :));
    end
       
    fclose(fid);
    ret = true;
end