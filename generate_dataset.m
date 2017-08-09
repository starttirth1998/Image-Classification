clear;
oldFolder = cd('~/CVIT/Image_Classification/Dataset/Cars_Val');

imagefiles1 = dir('*.png');
imagefiles2 = dir('*.jpg');
imagefiles3 = dir('*.jpeg');

imagefiles = [imagefiles1 ; imagefiles2 ; imagefiles3];

cnt = 1;
nfiles = length(imagefiles);
for i=1:nfiles
   disp(i);
   currentfilename = imagefiles(i).name;
   currentimage = imread(currentfilename);
   [r,c,no] = size(currentimage);
   if(no == 3)
       [fim,X_cars_val{cnt}] = vl_sift(single(rgb2gray(currentimage)));
   else
       [fim,X_cars_val{cnt}] = vl_sift(single(currentimage));
   end
   y_cars_val{cnt} = 5;
   images_cars_val{cnt} = currentimage;
   cnt = cnt + 1;
end

cd(oldFolder);
save('Cars_val.mat','X_cars_val','y_cars_val','images_cars_val');
