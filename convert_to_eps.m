function [number_image] = convert_to_eps(image_folder_base)
% this script is for converting the png or jpg file to eps file while
% walking all the files within the subfolders
% The input of this script is the path root of the folders
% The output is the number of images converted

content_list = dir(image_folder_base);
file_list1 = dir(fullfile(image_folder_base, '*.png'));
file_list2 = dir(fullfile(image_folder_base, '*.jpg'));
file_list3 = dir(fullfile(image_folder_base, '*.jpeg'));
file_list = [file_list1, file_list2, file_list3];
number_image = size(file_list, 1);

folder_list = content_list([content_list.isdir] & ~strncmpi('.', {content_list.name}, 1));

% convert the image in the current folder
for i = 1:size(file_list)
    name = file_list(i).name;
    index = strfind(name, '.');
    name_withoutformat = name(1:index-1);
    path = fullfile(image_folder_base, name);
    image = imread(path);
    imshow(image);
    savename = fullfile(image_folder_base, name_withoutformat);
    print(sprintf('%s.eps', savename), '-depsc');
end

% walk in the subfolders
for i = 1:size(folder_list, 1)
    subfolder_path = fullfile(image_folder_base, folder_list(i).name);
    number_image_temp = convert_to_eps(subfolder_path);
    number_image = number_image + number_image_temp;
end

end