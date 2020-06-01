
clear all; close all; clc; 
% See 
addpath('./testing')                          % must include this line
p = 3/2;q = 2;r = .5;                         % p, q, r
lim = 500;                                     % limit (max eigemvalue)
swap = true;                                  % swap intensities
k=17;K=18;                                    % set k and K 

% specify which directory under res_images to store the results in
filename='real_example';reps=5;
% specify the relative path to image (please store new images under
path = './sample_images/crows/original.jpg'; 

[TA, CA] = run_segmentation('kmeans', path, swap, p, q, r, lim, k, K, ...
    'brain');