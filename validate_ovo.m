clear;
load('~/CVIT/Image_Classification/Dataset/Bikes_test.mat');
load('~/CVIT/Image_Classification/Dataset/Airplane_test.mat');
load('~/CVIT/Image_Classification/Dataset/Ships_test.mat');
load('~/CVIT/Image_Classification/Dataset/Helicopters_test.mat');
load('~/CVIT/Image_Classification/Dataset/Buses_test.mat');
load('~/CVIT/Image_Classification/Dataset/Cars_test.mat');
load('~/CVIT/Image_Classification/Dataset/cluster.mat');
%load('~/CVIT/Image_Classification/Dataset/Final_Model.mat');
load('Model_ovo.mat');

X_cell = [X_bikes_test, X_airplane_test, X_ships_test,...
        X_helicopters_test, X_buses_test, X_cars_test];
y_cell = [y_bikes_test, y_airplane_test, y_ships_test,...
        y_helicopters_test, y_buses_test, y_cars_test];

%X_cell = [X_bikes_test];
%y_cell = [y_bikes_test];

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

label = zeros(length(model_ovo),length(X_cell));

for k1=1:length(model_ovo)
    [label(k1,:), accuracy, prob_values] =...
        svmpredict(y', X, model_ovo{1,k1},'-b 1');
end

%predict_label = zeros(length(X_cell));

for k1 = 1:length(X_cell)
    [M,I] = max(histcounts(label(:,k1),6));
    predict_label(k1) = I-1;
end

[confusion_mat, order] = confusionmat(y,predict_label);

%confusion_mat = (y==predict_label);
%accuracy = sum(double(confusion_mat))/length(y)

accuracy = sum(diag(confusion_mat))./sum(sum(confusion_mat))
%precision = diag(confusion_mat)./sum(confusion_mat,2);

%recall = diag(confusion_mat)./sum(confusion_mat,1)';