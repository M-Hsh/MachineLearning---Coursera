function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%vectorized cost function
temp_theta = theta(2 : end);
J = 1 / (2 * m) * sum(((X * theta) - y) .^ 2) + lambda / (2 * m) * sum(temp_theta .^ 2);

%non-vectorized grad computation
for i = 1 : m
    for j = 2 : size(theta)
        grad(j) += (1 / m) * (X(i, :) * theta - y(i)) * X(i, j);
    endfor
endfor
grad += (lambda / m) * theta;
grad(1) = 0;
for i = 1 : m
    grad(1) += (1 / m) * ((X(i, :) * theta - y(i)) * X(i, 1));
endfor




% =========================================================================

grad = grad(:);

end
