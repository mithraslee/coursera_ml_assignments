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



H_Theta_X = X * theta;
h = sigmoid(H_Theta_X);

theta_no_first = [0 ; theta(2:size(theta), :)];
J = -(log(h)' * y + log(1 - h)' * (1 - y))/m + lambda * sum(theta_no_first .^ 2) / (2 * m);
% J = -sum(log(h)' * y + log(1 - sigmoid(H_Theta_X))' * (1 - y))/m;

T = h - y;
% for i = 1:size(grad),
%     grad(i) = sum(T .* X(:,i)) / m;
% end

grad = (X' * T)/m + (lambda * theta_no_first)/m;
% grad(1) = sum(T .* X(:,1)) / m;

% =============================================================

end
