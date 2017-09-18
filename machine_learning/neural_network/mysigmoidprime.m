function gradient = mysigmoidprime(output)

% input is the matrix, which contains all linear combination from previous layer
% output is the matrix, by passing the input into the sigmoid function

	gradient = output.*(1-output);

end