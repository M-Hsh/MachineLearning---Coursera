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

%compute costFunction 
for i = 1:m
  h = sigmoid(X(i, :) * theta);
  J += (1 / m) *((-1) * y(i) * log(h) - (1 - y(i)) * log(1 - h));
endfor
for j = 1: size(grad)
  J += (lambda / 2 * m) * (theta(j) ^ 2);
endfor
%compute partial derivatives
grad(1) = 0;
for i = 1:m
  h = sigmoid(X(i, :) * theta);
  grad(1) += (1 / m) * (h - y(i));
endfor

for j = 2: size(grad)
  for i = 1:m
    h = sigmoid(X(i, :) * theta);
    grad(j) += (1 / m) * (h - y(i)) * X(i, j);
  endfor
  grad(j) += (lambda / m) * theta(j);
endfor
% =============================================================
end
