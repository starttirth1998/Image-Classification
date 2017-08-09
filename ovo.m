clear;
load('~/CVIT/Image_Classification/Dataset/Bikes_train.mat');
load('~/CVIT/Image_Classification/Dataset/Airplane_train.mat');
load('~/CVIT/Image_Classification/Dataset/Ships_train.mat');
load('~/CVIT/Image_Classification/Dataset/Helicopters_train.mat');
load('~/CVIT/Image_Classification/Dataset/Buses_train.mat');
load('~/CVIT/Image_Classification/Dataset/Cars_train.mat');
load('~/CVIT/Image_Classification/Dataset/cluster.mat');

dataset{1} = X_bikes_train;dataset{2} = X_airplane_train;dataset{3} = X_ships_train;
dataset{4} = X_helicopters_train;dataset{5} = X_buses_train;dataset{6} = X_cars_train;

actual{1} = y_bikes_train;actual{2} = y_airplane_train;actual{3} = y_ships_train;
actual{4} = y_helicopters_train;actual{5} = y_buses_train;actual{6} = y_cars_train;

cnt = 0;
for k1=1:length(dataset)-1
    for k2=k1+1:length(dataset)
        cnt = cnt + 1;
        X_cell = [dataset{k1}, dataset{k2}];
        y_cell = [actual{k1}, actual{k2}];

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

        model_ovo{cnt} = svmtrain(y', X, '-t 0 -c 500 -g 4 -b 1');

        [predict_label, accuracy, prob_values] =...
           svmpredict(y', X, model_ovo{cnt},'-b 1');
    end
end

save('Model_ovo.mat','model_ovo');