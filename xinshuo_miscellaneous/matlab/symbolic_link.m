% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function create a symbolic link for all systems
function symbolic_link(link, target)
    if ispc()	% check if windows platform
        system(sprintf('mklink /J %s %s', link, target)); 
    else 
        system(sprintf('ln -s %s %s', link, target)); 
    end
end
