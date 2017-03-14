
function ret = visualize_pts_folder(folder_path)
    assert(ischar(folder_path), 'The input folder is not valid.');
    subfolder_list = get_subfolder_list(folder_path);
    assert(~isempty(subfolder_list), 'No camera subfolder under the given folder found.');
    for i = 1:length(subfolder_list)
        image_path = sprintf('%s\\image0000.png', subfolder_list{i});
        file_path = sprintf('%s\\00000.pose', subfolder_list{i});
        visualize_pts_single_file(image_path, file_path);
    end
    ret = true;
end
