clear;
load('~/CVIT/Image_Classification/Dataset/Bikes_train.mat');
load('~/CVIT/Image_Classification/Dataset/Airplane_train.mat');
load('~/CVIT/Image_Classification/Dataset/Ships_train.mat');
load('~/CVIT/Image_Classification/Dataset/Helicopters_train.mat');
load('~/CVIT/Image_Classification/Dataset/Buses_train.mat');
load('~/CVIT/Image_Classification/Dataset/Cars_train.mat');

ele = 10;
X1 = [X_bikes_train{1,:}, X_airplane_train{1,:}, X_ships_train{1,:},...
        X_helicopters_train{1,:}, X_buses_train{1,:}, X_cars_train{1,:}];

ncluster = 100;

[idx,C] = kmeans(double(X1'),ncluster,'Maxiter',100,'Display','iter');

save('cluster.mat','idx','C');
