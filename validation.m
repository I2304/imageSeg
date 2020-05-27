clear all; close all; clc; 

% Example: 
evaluate_segmentation(3/2, 2, 1/2, 60, 3);  

% Takes in: 
%  P: the value of p to be used in the normalization
%  Q: the value of q to be used in the normalization
%  R: the value of r to be used in the normalization 
%  maxL: the maximum magnitude eigenvalue (e.g., 40) to be solved for
%  index: the id of the image (clean images: ids 1-4; noisy images: todo)
function [l] = evaluate_segmentation(P, Q, R, maxL, id)
    global M   
    global rho_matrix
    global q 
    global pr 
    epsilon = 10^(-3);
    % LOAD & PRE-PROCESS IMAGE --------------------------------------------
    [img, truth, num_clusters] = test_img(id);
    K = num_clusters-1; 
    figure(1)
    a1 = subplot(1, 3, 1);
    imagesc(img)
    colormap(a1, gray);
    c = colorbar; 
    ylabel(c, '[0, 1] Intensity', 'Interpreter', 'Latex', 'Fontsize', 14)
    xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 14)
    ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 14)
    title ('Original Image', 'Interpreter', 'Latex', 'Fontsize', 14)
    a2 = subplot(1, 3, 2);
    imagesc(truth)
    colormap(a2, jet); 
    c = colorbar; 
    ylabel(c, 'Cluster Index', 'Interpreter', 'Latex', 'Fontsize', 14)
    xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 14)
    ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 14)
    title ('Ground Truth', 'Interpreter', 'Latex', 'Fontsize', 14)
    % SET GLOBAL VARIABLES ------------------------------------------------
    rho_matrix = img;
    M = length(img); 
    q = Q; 
    pr = P + R;
    % CREAT EVP MODEL -----------------------------------------------------
    model = construct_laplacian_model();
    results = solve_evp(model,[-epsilon, maxL],epsilon);
    l = results.Eigenvalues;
    % EIGENFUNCTION PLOTS -------------------------------------------------
    [a, ~] = numSubplots(K);
    for i = 1:K
        figure(2)
        subplot(a(1), a(2), i)
        plot_varphi(i, results, R)
    end
    figure(2); sgtitle('Eigenfunctions $\varphi_i$', ...
        'Interpreter', 'Latex', 'Fontsize', 14);
    % RETRIEVE EMBEDDING --------------------------------------------------
    [uData, ~] = get_embedding(results, K, R);
    % RUN KMEANS ON EMBEDDING ---------------------------------------------
    figure(1)
    a3 = subplot(1, 3, 3);
    cluster(uData, truth, K, num_clusters, a3, 10);
    figure(3)
    qualities = 1:15;
    for i = 1:15
        a3 = subplot(3, 5, i);
        qualities(i) = cluster(uData, truth, K, num_clusters, a3, 1);
    end
    s = strcat('Highest Segmentation Accuracy (best of 10) was ', ...
        {' '}, num2str(mean(qualities)), ...
        ' for ($p$, $q$, $r$) = (', ...
        num2str(P), ',', num2str(Q),',', num2str(R), ')');
    sgtitle(s, 'Interpreter', 'Latex', 'Fontsize', 14);
end
% Plots the final segmentation using kmeans on the embedding
function quality = cluster(data, truth, K, num_clusters, a3, reps)
    X = reshape(data, [(length(data))^2, K]);
    clusters = reshape(kmeans(X, num_clusters, 'Replicates', reps), ...
        [(length(data)), (length(data))]); 
    quality = compute_quality(clusters, truth);
    imagesc(clusters)
    colormap(a3, jet); 
    c = colorbar; 
    ylabel(c, 'Cluster Index', 'Interpreter', 'Latex', 'Fontsize', 14)
    s = strcat('Segmentation (Accuracy = ', num2str(quality), ')');
    title(s, 'Interpreter', 'Latex', 'Fontsize', 14)
    xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 14)
    ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 14)
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

% Plots eigenfunction u_i
function plot_varphi(i, results, r)
    u = getu(i, results, r); 
    imagesc(u)
    colorbar
    s = strcat('Plot of $\varphi_', num2str(i), '$ mapped to image pixels');
    title(s,'Interpreter', 'Latex', 'Interpreter', 'Latex', ...
        'Fontsize', 14)
    xlabel('pixel vertical coordinate', 'Interpreter', 'Latex', ...
        'Fontsize', 14)
    ylabel('pixel horizontal coordinate', 'Interpreter', 'Latex', ...
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

function quality = compute_quality(clusters, truth)
    confusion_matrix=confusionmat(...
        reshape(truth, [size(truth, 1)^2, 1]),...
        reshape(clusters, [size(clusters, 1)^2, 1]));
    cost_matrix = make_cost_matrix(confusion_matrix); 
    [assignment, ~] = munkres(cost_matrix);
    cpy = clusters; 
    for i = 1:length(assignment)
        for j = 1:length(assignment)
            if assignment(i, j) == 1
             cpy(clusters == j) = i; 
            end
        end
    end
    quality = sum(sum(truth==cpy))/(length(cpy))^2;
end

% Compute cost matrix
function cost_matrix = make_cost_matrix(confusion_matrix)
    s = max(confusion_matrix); 
    cost_matrix = -confusion_matrix + s; 
end