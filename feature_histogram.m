clear;
load('~/CVIT/Image_Classification/Dataset/Bikes_train.mat');
load('~/CVIT/Image_Classification/Dataset/Airplane_train.mat');
load('~/CVIT/Image_Classification/Dataset/Ships_train.mat');
load('~/CVIT/Image_Classification/Dataset/Helicopters_train.mat');
load('~/CVIT/Image_Classification/Dataset/Buses_train.mat');
load('~/CVIT/Image_Classification/Dataset/Cars_train.mat');
load('~/CVIT/Image_Classification/Dataset/cluster.mat');

X_cell = [X_bikes_train, X_airplane_train, X_ships_train,...
        X_helicopters_train, X_buses_train, X_cars_train];
y_cell = [y_bikes_train, y_airplane_train, y_ships_train,...
        y_helicopters_train, y_buses_train, y_cars_train];

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

%model = svmtrain(y', [(1:length(y))' X*X'], '-t 2 -c 0.03 -g 0.07 -b 1');
model = ovrtrain(y', X, '-c 2 -g 4');

%model_ovo = svmtrain(y', [(1:length(y))' X], '-c 0.0001 -g 1');


%Code for one-vs-all classification
[predict_label, accuracy, prob_values] =...
    ovrpredict(y', X, model);

%[predict_label, accuracy, prob_values] =...
%   svmpredict(y', [(1:length(y))' X], model_ovo);

save('Model.mat','model');