clear;
load('~/CVIT/Image_Classification/Dataset/Bikes_test.mat');
load('~/CVIT/Image_Classification/Dataset/Airplane_test.mat');
load('~/CVIT/Image_Classification/Dataset/Ships_test.mat');
load('~/CVIT/Image_Classification/Dataset/Helicopters_test.mat');
load('~/CVIT/Image_Classification/Dataset/Buses_test.mat');
load('~/CVIT/Image_Classification/Dataset/Cars_test.mat');
load('~/CVIT/Image_Classification/Dataset/cluster.mat');
load('~/CVIT/Image_Classification/Dataset/Final_Model.mat');
%load('Model.mat');

X_cell = [X_bikes_test, X_airplane_test, X_ships_test,...
        X_helicopters_test, X_buses_test, X_cars_test];
y_cell = [y_bikes_test, y_airplane_test, y_ships_test,...
        y_helicopters_test, y_buses_test, y_cars_test];

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
addpath('~/CVIT/LIBSVM-multi-classification-master/');

%model = svmtrain(y', [(1:length(y))' X*X'], '-c 1 -g 0.07 -b 1');

%Code for one-vs-all classification
[predict_label, accuracy, prob_values] =...
    ovrpredict(y', X, model);

%[predict_label, accuracy, prob_values] =...
%   svmpredict(y', X, model_ovo);

[confusion_mat, order] = confusionmat(y',predict_label);

precision = diag(confusion_mat)./sum(confusion_mat,2);

recall = diag(confusion_mat)./sum(confusion_mat,1)';