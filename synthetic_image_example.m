clear all; close all; clc; 
addpath('./testing')                          % must include this line
p = 1.5;q = 2;r = .5;                         % p, q, r
lim = 60;                                     % limit (max eigemvalue)
% specify which directory under res_images to store the results in
filename='synthetic_example';reps=5; 
swap = true; 
[spec, truth, K] = test_img(4, true, .25);   % img 3, noise = .25
% [TA, CA] = run_segmentation('kmeans', spec, swap, p, q, r, ...
%     lim, K-1, K, filename, truth)
imagesc(truth)
colormap(jet)
colorbar
title('Ground Truth', 'Interpreter', 'Latex', 'Fontsize', 14)