%% Block Proximal Point (BPP) Algorithm for Training DNNs (1-layer Network)
clear all
close all
clc

addpath Algorithms Tools

disp('MLP with Three Hidden Layers using the MNIST dataset')

rng('default');
seed = 10;
rng(seed);
fprintf('Seed = %d \n', seed)

% read in MNIST dataset into Matlab format if not exist
if exist('mnist.mat', 'file')
    mnist = load('mnist.mat');
else
    disp('reading in MNIST dataset into Matlab format')
    addpath mnist-matlab
    convertMNIST
    mnist = load('mnist.mat');
end

% train data and labels
[x_d1,x_d2,x_d3] = size(mnist.training.images);
x_train = reshape(mnist.training.images,x_d1*x_d2,x_d3); % train data
% x_train = gpuArray(x_train);
y_train = mnist.training.labels; % labels
% y_train = gpuArray(y_train);

%% Extract Classes 
num_classes = 10; % choose the first num_class classes in the MNIST dataset for training
X = [y_train';x_train];
[~,col] = find(X(1,:) < num_classes);
X = X(:,col);
[~,N] = size(X);
X = X(:,randperm(N)); % shuffle the training dataset
x_train = X(2:end,:);
y_train = X(1,:)';
clear X

y_one_hot = ind2vec((y_train'+1));
[K,N] = size(y_one_hot);
[d,~] = size(x_train);

%% Test data
% read in test data and labels
[x_test_d1,x_test_d2,x_test_d3] = size(mnist.test.images);
x_test = reshape(mnist.test.images,x_test_d1*x_test_d2,x_test_d3); % test data
y_test = mnist.test.labels; % labels

X_test = [y_test';x_test];
[~, col_test] = find(X_test(1,:) < num_classes);
X_test = X_test(:,col_test);
[~,N_test] = size(X_test);
X_test = X_test(:,randperm(N_test,N_test)); % shuffle the test dataset
x_test = X_test(2:end,:);
y_test = X_test(1,:)';
clear X_test

y_test_one_hot = ind2vec((y_test'+1));
[~,N_test] = size(y_test_one_hot);

%% Visual data samples
% figure;
% for i = 1:100
%     subplot(10,10,i)
%     img{i} = reshape(x_train(:,i),x_d1,x_d2);
%     imshow(img{i})
% end
% 
% close all

%% Main Algorithm 1 (Proximal Point)
% Initialization of parameters 
d0 = d; d1 = 800; d2 = K;  % Layers: input + hidden + output
W1 = 0.01*randn(d1,d0); b1 = 0.1*ones(d1,1); 
W11 = W1; b11 = b1;

W2 = 0.01*randn(d2,d1); b2 = 0.1*ones(d2,1); 
W22 = W2; b22 = b2;

W3 = eye(K); b3 = zeros(K,1);


indicator = 1; % 0 = sign; 1 = ReLU; 2 = tanh; 3 = sigmoid

switch indicator
    case 0 % sign (binary)
        X1 = sign(W1*x_train+b1); X2 = sign(W2*X1+b2); %a3 = sign(W3*a2+b3);  
	case 1 % ReLU
        X1 = max(0,W1*x_train+b1); X2 = max(0,W2*X1+b2); %X3 = max(0,W3*X2+b3); X4 = max(0,W4*X3+b4);  
	case 2 % tanh
        X1 = tanh_proj(W1*x_train+b1); X2 = tanh_proj(W2*X1+b2); %a3 = tanh_proj(W3*a2+b3);
	case 3 % sigmoid
        X1 = sigmoid_proj(W1*x_train+b1); X2 = sigmoid_proj(W2*X1+b2); %a3 = sigmoid_proj(W3*a2+b3);
end
X11 = X1; X22 = X2; %X33 = X3; X44 = X4;
a1star = zeros(d1,N); a2star = zeros(d2,N); %a3star = zeros(d3,N); 
%u1 = zeros(d1,N); u2 = zeros(d2,N); u3 = zeros(d3,N); 

gamma = 1; gamma0 = 1;
gamma1 = gamma; gamma2 = gamma; gamma3 = gamma; gamma4 = gamma0; gamma5 = gamma0;
 alpha = 1; 

beta = 0.9;
omega = 0.01;

t = 0.1;

% niter = input('Number of iterations: ');
niter = 100;
loss1 = zeros(niter,1);
loss2 = zeros(niter,1);
accuracy_train = zeros(niter,1);
accuracy_test = zeros(niter,1);
time1 = zeros(niter,1);
X2 = y_one_hot;

% Iterations
for k = 1:niter
    tic
    
    t = 1;
    while t<60000
        X1(:,t:t+499) = max(W1*x_train(:,t:t+499)+b1,0);
        X11(:,t:t+499) = max(W11*x_train(:,t:t+499)+b11,0);
    [W22,W2] = UpdateWW(X1(:,t:t+499),X2(:,t:t+499),W22,W2,b2,alpha,omega,k);
    [b22,b2] = UpdateBB(X1(:,t:t+499),X2(:,t:t+499),W2,b22,b2,alpha,omega,k);
    [X11(:,t:t+499), X1(:,t:t+499)] = UpdateX(x_train(:,t:t+499),X11(:,t:t+499),X1(:,t:t+499),X2(:,t:t+499),W1,W2,b1,b2,alpha,gamma1,gamma2,omega,k);
    [W11,W1] = UpdateWW(x_train(:,t:t+499),X1(:,t:t+499),W11,W1,b1,alpha,omega,k);
    [b11,b1] = UpdateBB(x_train(:,t:t+499),X1(:,t:t+499),W1,b11,b1,alpha,omega,k);
    % adaptive momentum and update
%    [W1,b1,beta7] = AdaptiveWb1_3(lambda,gamma1,x_train,a1,W1,W1star,b1,b1star,beta7,t);
     t = t+500;
    end
    % Training accuracy
    switch indicator
    case 1 % ReLU
        X1_train = max(0,W1*x_train+b1);
        X2_train = max(0,W2*X1_train+b2);
        %X3_train = max(0,W3*X2_train+b3);
        %X4_train = max(0,W4*X3_train+b4);
%        a3_train = max(0,W3*a2_train+b3);
    end
    [~,pred] = max(X2_train,[],1);
    
    % Test accuracy
    switch indicator
        case 1 % ReLU
        x1_test = max(0,W1*x_test+b1); 
        x2_test = max(0,W2*x1_test+b2); 
        %x3_test = max(0,W3*x2_test+b3); 
        %x4_test = max(0,W4*x3_test+b4); 
    end
    [~,pred_test] = max(x2_test,[],1);
    
    loss1(k) = gamma0*norm(max(W2*X1+b2,0)-y_one_hot,'fro')^2;
    loss2(k) = loss1(k) + gamma1*norm(max(W1*x_train+b1,0)-X1,'fro')^2;
    accuracy_train(k) = sum(pred'-1 == y_train)/N;
    accuracy_test(k) = sum(pred_test'-1 == y_test)/N_test;
    time1(k) = toc;
    fprintf('epoch: %d, squared loss: %f, total loss: %f, training accuracy: %f, validation accuracy: %f, time: %f\n',k,loss1(k),loss2(k),accuracy_train(k),accuracy_test(k),time1(k));
  
end


fprintf('squared error: %f\n',loss1(k))
fprintf('sum of inter-layer loss: %f\n',loss2(k)-loss1(k))
%disp(full(cross_entropy(y_one_hot,a2,V,c)))


%% Plots
figure;
graph1 = semilogy(1:niter,loss1,1:niter,loss2);
set(graph1,'LineWidth',1.5);
legend('Squared loss','Total loss');
ylabel('Loss')
xlabel('Epochs')
title('Three-layer MLP')

figure;
graph2 = semilogy(1:niter,accuracy_train,1:niter,accuracy_test);
set(graph2,'LineWidth',1.5);
% ylim([0.85 1])
legend('Training accuracy','Test accuracy','Location','southeast');
ylabel('Accuracy')
xlabel('Epochs')
title('Three-layer MLP')

%% Training error
switch indicator
    case 1 % ReLU
        a1_train = max(0,W1*x_train+b1);
        a2_train = max(0,W2*a1_train+b2);
        %a3_train = max(0,W3*a2_train+b3);
        %a4_train = max(0,W4*a3_train+b4);
    case 2 % tanh
        a1_train = tanh_proj(W1*x_train+b1);
        a2_train = tanh_proj(W2*a1_train+b2);
        a3_train = tanh_proj(W3*a2_train+b3);
    case 3 % sigmoid
        a1_train = sigmoid_proj(W1*x_train+b1);
        a2_train = sigmoid_proj(W2*a1_train+b2);
        a3_train = sigmoid_proj(W3*a2_train+b3);
end

[~,pred] = max(a2_train,[],1);
pred_one_hot = ind2vec(pred);
accuracy_final = sum(pred'-1 == y_train)/N;
fprintf('Training accuracy using output layer: %f\n',accuracy_final);
% error = full(norm(pred_one_hot-y_one_hot,'fro')^2/(2*N));
% fprintf('Training MSE using output layer: %f\n',error);

%% Test errors
switch indicator
    case 1 % ReLU
        a1_test = max(0,W1*x_test+b1); 
        a2_test = max(0,W2*a1_test+b2); 
        %a3_test = max(0,W3*a2_test+b3); 
        %a4_test = max(0,W4*a3_test+b4); 
    case 2 % tanh
        a1_test = tanh_proj(W1*x_test+b1); 
        a2_test = tanh_proj(W2*a1_test+b2); 
        a3_test = tanh_proj(W3*a2_test+b3);
    case 3 % sigmoid
        a1_test = sigmoid_proj(W1*x_test+b1); 
        a2_test = sigmoid_proj(W2*a1_test+b2); 
        a3_test = sigmoid_proj(W3*a2_test+b3);
end


[~,pred_test] = max(a2_test,[],1);
pred_test_one_hot = ind2vec(pred_test);
accuracy_test_final = sum(pred_test'-1 == y_test)/N_test;
fprintf('Test accuracy using output layer: %f\n',accuracy_test_final);
% error_test = full(norm(pred_test_one_hot-y_test_one_hot,'fro')^2/(2*N_test));
% fprintf('Test MSE using output layer: %f\n',error_test);