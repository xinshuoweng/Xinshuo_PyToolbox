% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input number is a positive integer
function valid = isNonNegativeInteger(number) 
valid = isInteger(number) && ...			% is integer
        number >= 0;                      	% is positive
end