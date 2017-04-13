% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function creates data in hdf5 format from a image path 

% input parameter
%   data_src:       where the data is
%   label_src:      where the label is
%   save_dir:       where to store the hdf5 data
%   batch_size:     how many image to store in a single hdf file
%   ext_filder:     what format of data to use for generating hdf5 data 

function [num_hdf5, num_data] = generate_hdf5(save_dir, data_src, batch_size, ext_filter, label_src, function_name)
    print(function_name)
    % parse input data
    if ~exist('ext_filter', 'var')
        ext_filter = 'png';
    end

    if isFolder(data_src)
        filepath = file_abspath();
        datalist_name = 'datalist.txt';
        cmd = sprintf('th %s/../file_io/generate_list %s %s %s', filepath, data_src, datalist_name, ext_filter);
        system(cmd);    % generate data list
        [datalist, num_data] = load_list_from_file(data_src);
    elseif isFile(data_src)
        [datalist, num_data] = load_list_from_file(data_src);   
    else
        assert(false, 'data source format is not correct.');
    end

    % parse input label
    if ~exist('label_src', 'var') || isempty(label_src)
        labellist = {};
    elseif isFile(label_src)
        [labellist, num_label] = load_list_from_file(label_src);   
    else
        assert(false, 'label source format is not correct.');
    end

    assert(num_data == num_label, 'number of data and label is not equal.');
    assert(isFile(save_dir), 'save path should be a folder to save all hdf5 files')
    mkdir_is_missing(save_dir);

    if ~exist('batch_size', 'var')
        batch_size = 1;
    end

    % MAX_LABEL = 40;
    % MIN_LABEL = 0;
    % BATCH_SIZE = 2;
    % IMAGE_WIDTH = 320;
    % IMAGE_HEIGHT = 240;
    % SIZE_IMAGE = [IMAGE_HEIGHT, IMAGE_WIDTH, 3];

    %% create hdf5 file for training data
    % parent_dir = '/mnt/sdc1/xinshuow/dataset/google_street/newYork2';
    % f = fopen(sprintf('%s/rotatedimagelistwithLabel.txt', parent_dir), 'r');
    % assert(f ~= -1, 'image with label file not found');
    % train_path = textscan(f, '%s', 'Delimiter', '\n');
    % train_path = train_path{1};

    % save configuration
    % save_dir = sprintf('%s/hdf5/batch%d', parent_dir, BATCH_SIZE);
    % if ~exist(save_dir, 'dir')
        % mkdir(save_dir)
    % end

    size_data = size(imread(datalist{1}));
    count_hdf = 1;
    for i = 1:num_data
        fprintf('%d/%d\n', i, num_data);
        img = im2double(imread(datalist{i}));    % [rows,col,channel,numbers], scale the image data to (0, 1)
        if batch_size > 1
            assert(isequal(size_data, size(img)), 'image size should be equal in each single hdf5 file.');
        end
        size_data = size(img);
        data(:,:,:, mod(i-1, batch_size)+1) = img;

        if ~isempty(labellist)
            labels(1, mod(i-1, batch_size)+1) = str2double(labellist{i});  
        end

        if mod(i, batch_size) == 0
            % preprocess
            data = data(:, :, [3, 2, 1], :); % from rgb to brg
            data = permute(data, [2 1 3 4]);     % permute to [cols, rows, channel, numbers]
            
            % write to hdf5 format
            h5create(sprintf('%s/siamese_mirror_%05d.hdf5', save_dir, count_hdf), '/data', size(data), 'Datatype', 'double');  
            h5write(sprintf('%s/siamese_mirror_%05d.hdf5', save_dir, count_hdf), '/data', data);

            if ~isempty(labellist)
                h5create(sprintf('%s/siamese_mirror_%05d.hdf5', save_dir, count_hdf),'/label', size(labels), 'Datatype', 'double');
                h5write(sprintf('%s/siamese_mirror_%05d.hdf5', save_dir, count_hdf), '/label', labels);
                labels = zeros([1, batch_size]);   
            end

            % data = zeros([size(img), batch_size]);
            count_hdf = count_hdf + 1;
        end
        
    end

    % size(trainData)
    % size(trainLabels)

end