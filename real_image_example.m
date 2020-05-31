
clear all; close all; clc; 
% See 
addpath('./testing')                          % must include this line
p = 1.5;q = 2;r = .5;                         % p, q, r
lim = 60;                                     % limit (max eigemvalue)
swap = true;                                  % swap intensities
k = 6;K=7;                                    % set k and K 
% specify which directory under res_images to store the results in
filename='real_example';reps=5;
% specify the relative path to image (please store new images under
% sample_images
path = './sample_images/usnccm/original.png'; 
swap = true; 
[TA, CA] = run_segmentation('kmeans', path, swap, p, q, r, lim, k, K, ...
    filename);