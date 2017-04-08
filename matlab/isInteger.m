% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input number is an integer
function valid = isInteger(number) 
valid = ~ischar(number) && ...            % is numeric
        isscalar(number) && ...           % is scalar
        (fix(number) == number); 		  % is integer in value
end