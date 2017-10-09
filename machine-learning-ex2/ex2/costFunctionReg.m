function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
[hey,n]=size(X);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


sum1=0;
sum2=0;
for i=[1:m],
	valx=X(i,:)';
	h=sigmoid(theta'*valx);
	sum1=sum1+  (y(i)*log(h))+ (1-y(i))*log(1-h);
end;


for j=[2:n],
	sum2=sum2+ (lambda*theta(j)*theta(j));
end;



sum1=(-1/m)*sum1;
sum2=(1/(2*m))*sum2;
J=sum1+sum2;



sum1=0.0;
sum2=0.0;
sum3=0.0;
for j=[1:n],
	for i=[1:m],
		valx=X(i,:)';
		h=sigmoid(theta'*valx);
		if j==1,
			sum1=sum1+ ((h-y(i))*X(i,j));
		else
			sum2=sum2+ ( (h-y(i))*X(i,j) );
		end;
	end;
	sum3=sum3+lambda*theta(j);
	if j==1,
		grad(j)=(1/m)*sum1;
	else
		grad(j)=(1/m)*(sum2+sum3);
	end;
	sum2=0.0;
	sum3=0.0;
end;







% =============================================================

end
