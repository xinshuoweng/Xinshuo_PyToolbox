% this function read a image and get point from users and save them to
% .pose file with specific path

function nbytes = getPointFromImage(image_path, number_point, save_path)
    image = imread(image_path);
    figure;
    imshow(image);
    point = ginput(number_point);

    fileID = fopen(save_path, 'w');
    nbytes = 0
    for i = 1:number_point
        nbytes = nbytes + fprintf(fileID, '%05.5f ', point);     
    end
    fprintf(fileID, '\n');     
end