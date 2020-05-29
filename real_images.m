clear all; close all; clc;
addpath('.\fuzme_matlab')

%% Main execution block

% Example segmentations
% WORKS BEST WITH num_clusters = K+1

% All test cases here are for the normalized laplacian (3/2, 2, 1/2) = (P,
% Q, R). These parameters can be tuned

% [l, uData, vData] = get_segmentation('./test_cases/density_2_dots/original.png', ...
%     3/2, 2, 1/2, 2, 70, 3, 3, true);

[l, uData, vData, clusters] = get_segmentation('./real_images/brain/original.png', ...
    3/2, 2, 1/2, 4, 70, 5, false);

% [l, uData, vData] = get_segmentation('./test_cases/pattern_12_dots/original.png', ...
%     3/2, 2, 1/2, 11, 80, 12, 12, true);

% [l, uData, vData] = get_segmentation('./test_cases/usnccm_7_letters/original.png', ...
%     3/2, 2, 1/2, 6, 70, 7, 7, true);

% [l, uData, vData, clusters] = get_segmentation('./real_images/flower/original.jpg', ...
%     3/2, 2, 1/2, 4, 80, 5, false);




%% Segmentations Algorithm
% Takes in:
%  path: the (relative) path to an image
%  P: the value of p to be used in the normalization
%  Q: the value of q to be used in the normalization
%  R: the value of r to be used in the normalization
%  K: the number of eigenfunctions (e.g., 6) to be plotted
%  maxL: the maximum magnitude eigenvalue (e.g., 40) to be solved for
%  num_clusters_u: number of clusters when segmenting on u
%  num_clusters_v: number of clusters when segmenting on v
%  swap: if the segmentation is bad, consider switching swap to true
%        to swap the contrast of the image as this may improve the
%        calculation of the eigenfunctions
% Plots the specified range of eigenfunctions v_{minI}, ..., v_{maxI} to
% be plotted, and returns l (the list of eigenvalues identified). It also
% returns embeddings in terms of u and v.
function [l, uData, vData, clusters] = get_segmentation(path, P, Q, R, K, maxL, ...
    num_clusters, swap)
    dest = ['./image_results/' nextname('./image_results/figure_set','_1','')];
    mkdir(dest);
    global M
    global rho_matrix
    global q
    global pr
    epsilon = 10^(-3);
    % LOAD & PRE-PROCESS IMAGE --------------------------------------------
    img = mat2gray(imread(path));
    % Force image to be square
    img = img(1:min(size(img, 1), size(img, 2)), ...
        1:min(size(img, 1), size(img, 2)));
    if swap == true
        img = 1-(cast(img, 'double'));
    end
    % Make a small correction to zero terms (to avoid numerical issues)
    img(img == 0) = .01;
    figure(1)
    imagesc(img)
    colormap(gray)
    c = colorbar; 
    ylabel(c, '[0, 1] intensity', 'Interpreter', 'Latex', 'Fontsize', 14)
    xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 14)
    ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 14)
    title ('Original Image', 'Interpreter', 'Latex', 'Fontsize', 14)
    temp=[dest,filesep,'original_image.png'];
    saveas(gca,temp);
    % SET GLOBAL VARIABLES ------------------------------------------------
    rho_matrix = img;
    M = length(img);
    q = Q;
    pr = P + R;
    % CREATE EVP MODEL ----------------------------------------------------
    model = construct_laplacian_model();
    results = solve_evp(model,[-epsilon, maxL],epsilon);
    l = results.Eigenvalues;
    % EIGENFUNCTION PLOTS -------------------------------------------------
    [a, ~] = numSubplots(K);
    figure(2)
    for i = 1:K
        % Plot u_i
        subplot(a(1), a(2), i)
        plot_u(i, results, R)
    end
    temp=[dest,filesep,'eigenfunctions_', ...
        erase(num2str(P), '.'),'_', erase(num2str(Q), '.') ,...
        '_',erase(num2str(R), '.'),'.png'];
    [ax,h] = suplabel('Eigenfunctions', 't');
    set(h, 'Interpreter', 'Latex', 'Fontsize', 14);
    saveas(gca,temp);
    % RETRIEVE EMBEDDING & KMEANS -----------------------------------------
    [uData, vData] = get_embedding(results, K, R);
    figure(3)
    clusters = fuzzy_cluster(uData, K, num_clusters, ...
        ['Segmentation using $\{u_i\}$ : p = ' num2str(P) ' , q = ' num2str(Q) ' , r = ' num2str(R)])
    temp=[dest,filesep,'segmentation_', erase(num2str(P), '.'),...
        '_',erase(num2str(Q), '.'),'_',erase(num2str(R), '.'),'.png'];
    saveas(gca,temp);
end
% Plots the final segmentation using kmeans on the embedding
function[clusters] = cluster(data, K, num_clusters, s)
    X = reshape(data, [(length(data))^2, K]);
    clusters = reshape(kmeans(X, num_clusters), ...
        [(length(data)), (length(data))]);
    imagesc(clusters)
    title(s, 'Interpreter', 'Latex', 'Fontsize', 14)
    c = colorbar;
    colorTitleHandle = get(c,'Title');
    titleString = 'Cluster Index';
    set(colorTitleHandle ,'String',titleString, 'Interpreter', 'Latex', ...
        'Fontsize', 14);
    xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 14)
    ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 14)
end

% Plots the final segmentation using fuzzy kmeans on the embedding
function[clusters] =  fuzzy_cluster(data, K, num_clusters, s)
    X = reshape(data, [(length(data))^2, K]);
    out = run_fuzme(num_clusters, X, 2, 100, 1, 0.000001, 0.2, 10);
    clusters = reshape(out, [length(data), length(data), num_clusters]);
    for k = 1:num_clusters
        cluster_k = clusters(:, :, k);
        imagesc(cluster_k)
        title(s, 'Interpreter', 'Latex', 'Fontsize', 14)
        xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 14)
        ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 14)
        c = colorbar;
        c.Limits = [0, 1];
        colorTitleHandle = get(c,'Title');
        titleString = 'Cluster Probability';
        set(colorTitleHandle ,'String',titleString, 'Interpreter', 'Latex', ...
            'Fontsize', 14);
        figure(k + 3)
    end

end

% Computes the embedding of the image from the eigenfunctions
function [uData, vData] = get_embedding(results, K, r)
    global M
    global rho_matrix
    vData = zeros(M, M, K);
    uData = zeros(M, M, K);
    for k = 1:K
        [xq,yq] = meshgrid(linspace(0, 1, M));
        v = interpolateSolution(results,xq,yq,k);
        u = flip(reshape(v,size(xq)), 1).*((rho_matrix).^(r));
        vData(:, :, k) = flip(reshape(v,size(xq)), 1);
        uData(:, :, k) = u;
    end
end
%% Plotting helper functions
% Plots eigenfunction u_i
function plot_u(i, results, r)
    u = getu(i, results, r);
    imagesc(u)
    colorbar
    s = strcat('$u_', num2str(i), '$');
    title(s,'Interpreter', 'Latex', 'Interpreter', 'Latex', ...
        'Fontsize', 14)
    xlabel('pixel $i$', 'Interpreter', 'Latex', ...
        'Fontsize', 14)
    ylabel('pixel $j$', 'Interpreter', 'Latex', ...
        'Fontsize', 14)
end
% Takes in the results of the eigenvalue solver and returns corresponding
% eigenfunction u_i.
function u = getu(i, results, r)
    global M
    global rho_matrix
    [xq,yq] = meshgrid(linspace(0, 1, M));
    v = interpolateSolution(results,xq,yq,i);
    u = flip(reshape(v,size(xq)), 1).*((rho_matrix).^(r));
end

%% Eigensolver helper functions.
% Solves eigenvalue problem and runs error checking.
function results = solve_evp(model, range, epsilon)
    results = solvepdeeig(model,range);
    l = results.Eigenvalues; % stores eigenvalues
    if abs(l(1)) > epsilon
        disp('Uh oh, the first eigenvalue is nonzero. Check range.')
    end
end
% Computes a pde model for the L operator with zero neumann BC's,
% specified square geometry, and custom tolerance and mesh size.
function model = construct_laplacian_model()
    g = get_square_geometry();
    model = createpde();
    % Geometry specification
    geometryFromEdges(model,g);
    % Solver specificiation
    model.SolverOptions.AbsoluteTolerance = 5.0000e-025;
    model.SolverOptions.RelativeTolerance = 5.0000e-025;
    model.SolverOptions.ReportStatistics = 'on';
    % Homogeneous Neumann BCs
    applyBoundaryCondition(model,'neumann','Edge',1:4,'g',0,'q',0);
    % Coefficient specification
    specifyCoefficients(model,'m',0,'d',@dcoeffunction,...
        'c',@ccoeffunction,'a',0,'f',0);
    % Mesh size
    generateMesh(model,'Hmax',0.01)
end
% Computes a [0, 1] x [0, 1] square geometry.
function g = get_square_geometry()
    lowerLeft  = [0 , 0 ];
    lowerRight = [1 , 0 ];
    upperRight = [1 , 1];
    upperLeft =  [0 , 1];
    S = [3,4 lowerLeft(1), lowerRight(1), upperRight(1), upperLeft(1), ...
             lowerLeft(2), lowerRight(2), upperRight(2), upperLeft(2)];
    gdm = S';
    % Names
    ns = 'S';
    % Set formula
    sf = 'S';
    % Invoke decsg
    g = decsg(gdm,ns,sf');
end

%% Coefficient specification functions (must follow a very specific format)
% This function encodes the coefficient d, which in our formulation is
% \rho^(p+r).
function dmatrix = dcoeffunction(location,state)
    % Use global definition of rho_matrix
    global rho_matrix
    % Use global definition of p+r = pr
    global pr
    % Use global M (number of pixels)
    global M
    nr = numel(location.x);
    dmatrix = ones(1,nr);
    % Iterate over all nodes in location, and evaluate the value of
    % rho^(p+r) pointwise. Store the result in d_matrix.
    for index = 1:nr
        % Obtain position in location vector
        x = location.x(index);  % x location in mesh
        y = location.y(index);  % y location in mesh
        % Map (x, y) --> (i, j) pixel in image
        i = ceil(M*(1-y));      % i position in image
        j = ceil(M*x);          % j position in image
        % Correction for edge case
        if i == 0
            i = 1;
        end
        if j == 0
            j = 1;
        end
        % Store result
        dmatrix(index) = (rho_matrix(i,j))^(pr);
    end
end
% This function encodes the coefficient c, which in our formulation is
% \rho^(q).
function cmatrix = ccoeffunction(location,state)
    % Use global definition of rho_matrix
    global rho_matrix
    % Use global definition of p+r = pr
    global q
    % Use global M (number of pixels)
    global M
    nr = numel(location.x);
    cmatrix = ones(1,nr);
    % Iterate over all nodes in location, and evaluate the value of
    % rho^(p+r) pointwise. Store the result in d_matrix.
    for index = 1:nr
        % Obtain position in location vector
        x = location.x(index);  % x location in mesh
        y = location.y(index);  % y location in mesh
        % Map (x, y) --> (i, j) pixel in image
        i = ceil(M*(1-y));      % i position in image
        j = ceil(M*x);          % j position in image
        % Correction for edge case
        if i == 0
            i = 1;
        end
        if j == 0
            j = 1;
        end
        % Store result
        cmatrix(index) = (rho_matrix(i,j))^(q);
    end
end
