

function [theta, J_history] = gradientDescent(X, y, theta)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
num_iters = 1500;
J_history = zeros(num_iters, 1);

alpha = 0.01;
for iter = 1:400

h = sigmoid(X*theta);
sqrErrors1 = (h - y) * X(:,1);
sqrErrors2 = (h - y) * X(:,2);
sqrErrors3 = (h - y) * X(:,3);

temp1 = theta(1) - (1/m * sqrErrors1) * alpha;
temp2 = theta(2) - (1/m * sqrErrors2) * alpha;
temp3 = theta(2) - (1/m * sqrErrors3) * alpha;
%temp0 = theta(1) - (alpha * J);
%temp1 = theta(2) - (alpha * J);
%temp0 = theta(1) - alpha * J;
%temp1 = theta(2) - alpha * J;
%disp(J_history(iter));
theta(1) = temp1;
theta(2) = temp2;
theta(3) = temp3;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = costFunction(theta,X,y);

end

end
