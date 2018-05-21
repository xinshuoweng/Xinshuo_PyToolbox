% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

function generate_images_from_video(input_video, output_dir, debug_mode)
	if nargin < 3
		debug_mode = true;
	end
	
	if debug_mode
		assert(ischar(input_video), 'the input video is not correct.');
		assert(ischar(output_dir), 'the output directory is not correct.');
	end

	% assigning the name of sample avi file to a variable
	% videoname = '3.mp4';

	%reading a video file
	mov = VideoReader(input_video);
	mkdir_if_missing(output_dir);

	%getting no of frames
	numFrames = mov.NumberOfFrames;

	%setting current status of number of frames written to zero
	numFramesWritten = 0;
	time = tic;
	%for loop to traverse & process from frame '1' to 'last' frames
	for t = 1 : numFrames
		currFrame = read(mov, t);    %reading individual frames
		filename = sprintf('%010d.png', t);
		savepath_tmp = fullfile(output_dir, filename);
		imwrite(currFrame, savepath_tmp, 'png');   %saving as 'png' file

		% count the time
		elapsed = toc(time);
		remaining_str = string(py.timer.format_time(elapsed / t * (numFrames - t)));
		elapsed_str = string(py.timer.format_time(toc(time)));
		
		fprintf('Extracting frames: %d/%d, path: %s, EP: %s, ETA: %s\n', t, numFrames, filename, elapsed_str, remaining_str);
	    numFramesWritten = numFramesWritten + 1;
	end      

	progIndication = sprintf('Wrote %d frames to folder "%s"',numFramesWritten, output_dir);
	disp(progIndication);
end