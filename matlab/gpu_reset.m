% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function reset the gpu occupied by MATLAB, should
% be executed before gpu is used in MATLAB
function dummy = gpu_reset()
    try
        gpu = gpuDevice;
        clear gpu;
    catch
        disp('No GPU device found during startup');
    end
end