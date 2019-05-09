function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

predictions = theta' * X';
sqrErrors1 = (predictions - y') * X(:,1);
sqrErrors2 = (predictions - y') * X(:,2);
%sqrErrors = (predictions-y).^2; 
%J = 1/(2*m) * sum(sqrErrors*X'); 
%J = computeCost(X, y, theta);
temp0 = theta(1) - (1/m * sqrErrors1) * alpha;
temp1 = theta(2) - (1/m * sqrErrors2) * alpha;
%temp0 = theta(1) - (alpha * J);
%temp1 = theta(2) - (alpha * J);
%temp0 = theta(1) - alpha * J;
%temp1 = theta(2) - alpha * J;
%disp(J_history(iter));
theta(1) = temp0;
theta(2) = temp1;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
