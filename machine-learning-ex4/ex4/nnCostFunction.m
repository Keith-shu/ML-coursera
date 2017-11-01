function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X=[ones(m,1) X];
yr=zeros(m,num_labels);
for i = 1:m
    yr(i,y(i))=1;
end
for i = 1:m
    a1=X(i,:);
    a2=sigmoid(a1*Theta1');
    a2=[1 a2];
    a3=sigmoid(a2*Theta2');
    hx=a3';
    yi=yr(i,:);
    yi=yi';
    if size(yi)==size(hx)
        J=J+sum(-yi.*log(hx)-(1-yi).*log(1-hx));
    end
end
J=J/m;

theta1=Theta1(:,2:end).^2;
theta2=Theta2(:,2:end).^2;
theta1_all=theta1(:);
theta2_all=theta2(:);
temp=sum(theta1_all)+sum(theta2_all);
J=J+(lambda*temp)/(2*m);

Delta1=zeros(size(Theta1));
Delta2=zeros(size(Theta2));
Delta2=0;
for t =1:m
    a1=X(t,:);% 401x1
    z2=a1*Theta1'; %1x401x401x25=1x25
    a2=sigmoid(z2);
    a2=[1 a2]; %1x26
    z3=a2*Theta2'; %1x26x26x10=1x10
    a3=sigmoid(z3);%1x10
    hx=a3'; % 10x1
    yt=yr(t,:); % 1x10
    yt=yt'; %1x10
    delta3=hx-yt; % 10x1
    %delta_2=Theta2'*delta_3; %26x10x10x1=26x1
    %note:theta2 bias units should be ignored,u can't find this in ex4.pdf,so
    delta2=Theta2(:,2:end)'*delta3; % now we have 25x1 vector
    delta2=delta2.*sigmoidGradient(z2'); %25x1
    Delta1=Delta1+delta2*a1;%25x1x1x401=25x401
    Delta2=Delta2+delta3*a2;
end
Theta1_grad(:,1)=Delta1(:,1)/m;
Theta1_grad(:,2:end)=Delta1(:,2:end)/m+(lambda/m)*Theta1(:,2:end);
Theta2_grad(:,1)=Delta2(:,1)/m;
Theta2_grad(:,2:end)=Delta2(:,2:end)/m+(lambda/m)*Theta2(:,2:end);



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
