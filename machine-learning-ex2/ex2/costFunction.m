function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
alpha=1;
[hey,n]=size(X);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for i=[1:m],

	valx=X(i,:)';          % X vector for ith training examples
	J=J+ y(i)*log(sigmoid(theta'*valx)) + (1-y(i))*log(1-sigmoid(theta'*valx));
end;

J=(-1*J)/(m);




sum=0.0;
for j=[1:n],
	for i=[1:m],
		valx=X(i,:)';
		h=sigmoid(theta'*valx);
		sum=sum+ ( (h-y(i))*X(i,j) );
	end;
	grad(j)=(1/m)*sum;
	sum=0;
end;







% =============================================================

end
