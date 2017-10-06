% this function assumes the parallel computing toolbox
function gpu_id = auto_select_gpu()
    % deselects all GPU devices
    gpuDevice([]);
    
    maxFreeMemory = 0;
    for i = 1:gpuDeviceCount
        g = gpuDevice(i);
        freeMemory = g.FreeMemory();
        reset(g);
        fprintf('GPU %d: free memory %d\n', i, freeMemory);
        if freeMemory > maxFreeMemory
            maxFreeMemory = freeMemory;
            gpu_id = g;
%             gpu_handler = g;
        end
    end
    fprintf('Use GPU %d\n', 4 - gpu_id);
    
    % deselects all GPU devices
    gpuDevice([]);

end