% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes a file of points and an image as input
% then draw the points on top of the image as visualization
% this function assumer each row fo file has [x, y] coordinate
function img_with_pts = visualize_image_with_pts_from_file(img, pts_path)
	img = isImageorPath(img);
	fid = get_fileID_for_loading(pts_path);
	figure;
	imshow(img);
	hold on;

	tline = fgetl(fid);
	while ischar(tline)
		contents = strsplit(tline, ' ');
		plot(str2double(contents{1}), str2double(contents{2}), 'Marker', '*', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
		tline = fgetl(fid);
	end
	fclose(fid);
	img_with_pts = getframe;
	img_with_pts = img_with_pts.cdata;
end
