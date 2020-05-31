clear all; close all; clc; 
params = [...
    1, 1, 1; ...
    2, 1, 0; ...
    1.5, 1.0, .5; ...
];
addpath('./testing')
imgs = [1, 0; 1, .25; 2, 0; 2, .03];
tabs = zeros(length(params)*4, 9);
index = 1; 
for id = 1:length(imgs)
    for num = 1:length(params)
        close all force
        % Retrive image
        img = imgs(id, :); 
        [spec, truth, K] = test_img(img(1)+2, true, img(2)); 
        k = K-1; 
        % Set parameters; 
        row = params(num, :); 
        p = row(1); q = row(2); r = row(3); 
        lim = 50; 
        filename = ['/experiments/smallq/fig', num2str(img(1)), ...
            '_noise_', erase(num2str(img(2)), '.'), ...
            '/p_', erase(num2str(p), '.')...
            '_q_', erase(num2str(q), '.'), ...
            '_r_', erase(num2str(r), '.')];
        [avgTA, avgCA, stdTA, stdCA] = run_pqr_experiment(spec, p, q, ...
            r, lim, k, K, filename, truth, 12);
        tabs(index, 1) = img(1); 
        tabs(index, 2) = img(2);
        tabs(index, 3:5) = [p, q, r]; 
        tabs(index, 6:9) = [avgTA, avgCA, stdTA, stdCA];
        index = index + 1; 
    end
end
tabs(:, 6:9) = round(tabs(:, 6:9), 4);
writematrix(tabs,'./res_images/experiments/smallq/data.csv')