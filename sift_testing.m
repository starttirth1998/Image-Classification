oldFolder = cd('~/CVIT/Image_Classification/Airplane_Train/');

im = imread('image_0002.jpg');

[fim,dim] = vl_sift(single(rgb2gray(im)));

cd(oldFolder);