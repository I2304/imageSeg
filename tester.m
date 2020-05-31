clear all; close all; clc; 
p = 1.5;q = 2;r = .5;lim = 60;k = 6;K=7;filename='usnccm';reps=5;
addpath('./testing')
path = './sample_images/usnccm/original.png'; 
swap = true; 
% [spec, truth, K] = test_img(3, true, .25);
[TA, CA] = run_segmentation('kmeans', path, swap, p, q, r, lim, k, K, ...
    'testingtesting');