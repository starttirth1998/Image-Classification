clear;
load('~/CVIT/Image_Classification/Dataset/Bikes_val.mat');
load('~/CVIT/Image_Classification/Dataset/Airplane_val.mat');
load('~/CVIT/Image_Classification/Dataset/Ships_val.mat');
load('~/CVIT/Image_Classification/Dataset/Helicopters_val.mat');
load('~/CVIT/Image_Classification/Dataset/Buses_val.mat');
load('~/CVIT/Image_Classification/Dataset/Cars_val.mat');
load('~/CVIT/Image_Classification/Dataset/cluster.mat');
load('~/CVIT/Image_Classification/Dataset/Final_Model.mat');
%load('Model.mat');

X_cell = [X_bikes_val, X_airplane_val, X_ships_val,...
        X_helicopters_val, X_buses_val, X_cars_val];
y_cell = [y_bikes_val, y_airplane_val, y_ships_val,...
        y_helicopters_val, y_buses_val, y_cars_val];

ncluster = 100;

for i=1:length(X_cell)
    disp(i);
    dist = pdist2(double(X_cell{1,i}'),C);
    [M,cluster_number{i}] = min(dist,[],2);
    X(i,:) = histcounts(cluster_number{i},ncluster)./...
               sum(histcounts(cluster_number{i},ncluster));
    y(i) = y_cell{1,i};
    %pause;
end

addpath('~/CVIT/libsvm-3.22/matlab/');
addpath('~/CVIT/liblinear-ovo-2.11/');

%model = svmtrain(y', [(1:length(y))' X*X'], '-c 1 -g 0.07 -b 1');

%Code for one-vs-all classification
[predict_label, accuracy, prob_values] =...
    ovrpredict(y', X, model);

%[predict_label, accuracy, prob_values] =...
%   svmpredict(y', [(1:length(y))' X], model_ovo);

[confusion_mat, order] = confusionmat(y',predict_label);

precision = diag(confusion_mat)./sum(confusion_mat,2);

recall = diag(confusion_mat)./sum(confusion_mat,1)';