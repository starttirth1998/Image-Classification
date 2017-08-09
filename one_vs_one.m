%# load dataset
load fisheriris
[g gn] = grp2idx(species);                      %# nominal class to numeric
%# split training/testing sets
[trainIdx testIdx] = crossvalind('HoldOut', species, 1/3);
pairwise = nchoosek(1:length(gn),2);            %# 1-vs-1 pairwise models
svmModel = cell(size(pairwise,1),1);            %# store binary-classifers
predTest = zeros(sum(testIdx),numel(svmModel)); %# store binary predictions
%# classify using one-against-one approach, SVM with 3rd degree poly kernel
for k=1:numel(svmModel)
    %# get only training instances belonging to this pair
    idx = trainIdx & any( bsxfun(@eq, g, pairwise(k,:)) , 2 );
      %# train
      svmModel{k} = svmtrain(meas(idx,:), g(idx),'-s 0 -t 0');
      %# test
      predTest(:,k) = svmclassify(svmModel{k}, meas(testIdx,:));
  end
  pred = mode(predTest,2);   %# voting: clasify as the class receiving most votes
%# performance
cmat = confusionmat(g(testIdx),pred);
acc = 100*sum(diag(cmat))./sum(cmat(:));
fprintf('SVM (1-against-1):\naccuracy = %.2f%%\n', acc);
fprintf('Confusion Matrix:\n'), disp(cmat)