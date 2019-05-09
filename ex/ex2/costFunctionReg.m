function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X*theta);
part1 = -y' * log(h);
part2 = (1-y)' * log(1-h);
regu = (lambda / (2*m)) * sum(theta(2:size(theta)) .^ 2);
J = 1/m * sum((part1 - part2));
J = J + regu;

theta(1) = (X(:,1)' * (h - y)) / m;
theta(2) = ((X(:,2)' * (h - y)) / m) + (lambda / m) * (theta(2));
theta(3) = ((X(:,3)' * (h - y)) / m) + (lambda / m) * (theta(3));

grad = theta;

% =============================================================

end
