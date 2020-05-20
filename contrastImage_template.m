
% Please do not modify this file!

% This is an example snippet showing how to increase the contrast of an 
% image before feeding into the image segmentation algorithm. 

% Make a copy of this file to replace the path with the relative path 
% to the image file of your choice. This will bring up a window, where you
% can adjust the contrast as neccessary (click the half white/half black 
% circle). Once the contrast is high enough, save the image under a new 
% name (e.g., for the snipper below, I used density_contrasted.png as 
% the new file name). Only pass contrasted images into the image 
% segmentation algorithm. 

og = imread('./us_usnccm/density.png');
img = mat2gray(og);
img = img(1:min(size(img, 1), size(img, 2)), ...
    1:min(size(img, 1), size(img, 2)));
imtool(img)

