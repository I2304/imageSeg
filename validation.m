clear all; close all; clc; 

%% Test cases 
rho_1_dense = .01*ones(700); % Two squares. 
    rho_1_dense(200:400, 200:400) = 1; 
    rho_1_dense(500:600, 500:600) = 1; 
rho_1_truth = 1*ones(700); % Two squares. 
    rho_1_truth(200:400, 200:400) = 2; 
    rho_1_truth(500:600, 500:600) = 3; 
    
rho_2_dense = .01*ones(1000); % Multiple objects, nested
rho_2_truth = ones(1000); 
    for i = 1:1000
        for j = 1:1000
            if i <= 400 && j <= 400 && i >= 100 && j >= 100
                rho_2_dense(i, j) = 1; 
                rho_2_truth(i, j) = 2; 
            end
            if (i-700)^2+(j-700)^2 <= 8000
                rho_2_dense(i, j) = 1; 
                rho_2_truth(i, j) = 3; 
            end
            if (i-600)^2+(j-600)^2 <= 800
                rho_2_dense(i, j) = 1; 
                rho_2_truth(i, j) = 4; 
            end
            if i <= 300 && j <= 800 && i >= 200 && j >= 500 && i+500 <= j
                rho_2_dense(i, j) = 1;
                rho_2_truth(i, j) = 5;
            end
        end
    end
rho_2_dense(150:350, 150:350) = .01; 
rho_2_truth(150:350, 150:350) = 1; 
rho_2_dense(150:400, 200:230) = 1; 
rho_2_truth(150:400, 200:230) = 2; 
rho_2_dense(275:325, 275:325) = 1; 
rho_2_truth(275:325, 275:325) = 6; 
rho_2_dense(200:225, 275:300) = 1; 
rho_2_truth(200:225, 275:300) = 7; 
rho_2_dense(800:900, 100:200) = 1; 
rho_2_truth(800:900, 100:200) = 8; 
N = 10; 

% %% Main execution block 
accuracies = 1:N; 
for reps=1:N
    [l, clustersu, clustersv] = get_segmentation(rho_2_dense, 3/2, 3/2, 1/2, ...
        7, 70, 8, 8);
    accuracies(reps) = compute_quality(clustersu, 8, rho_2_truth);
end
mean(accuracies)

function quality = compute_quality(clusters, num_clusters, truth)
    % compute all permutations
    permutations = perms(1:num_clusters);
    % preallocate memory for errors
    acc = zeros(size(permutations, 1), 1);
    % loop over all permutations p
    for p = 1:size(permutations, 1)
        % make a copy of the clusters
        temp = clusters; 
        % consider the permutation p
        for i = 1:num_clusters
            temp(clusters == i) = permutations(p, i); 
        end
        acc(p) = sum(sum(temp==truth))/(length(temp))^2;
    end
    quality = max(acc);
end

%% Segmentation Algorithm
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
function [l, clustersu, clustersv] = get_segmentation(img, P, Q, R, K, maxL, ...
    num_clusters_u, num_clusters_v)
    global M   
    global rho_matrix
    global q 
    global pr 
    epsilon = 10^(-3);
    % LOAD & PRE-PROCESS IMAGE --------------------------------------------
    figure(1)
    imshow(img)
    title ('Original Image', 'Interpreter', 'Latex', 'Fontsize', 14)
    % SET GLOBAL VARIABLES ------------------------------------------------
    rho_matrix = img;
    M = length(img); 
    q = Q; 
    pr = P + R;
    % CREAT EVP MODEL -----------------------------------------------------
    model = construct_laplacian_model();
    % Plot the domain
    figure(2)
    plot_domain(model)
    % Calculate eigenvalues
    results = solve_evp(model,[-epsilon, maxL],epsilon);
    l = results.Eigenvalues;
    % EIGENFUNCTION PLOTS -------------------------------------------------
    [a, ~] = numSubplots(K);
    for i = 1:K
        % Plot v_i
        figure(3)
        subplot(a(1), a(2), i)
        plot_v(i, results, model)
        % Plot u_i
        figure(4)
        subplot(a(1), a(2), i)
        plot_u(i, results, R)
    end
    figure(3); sgtitle('Transformed eigenvectors $v$', 'Interpreter', 'Latex');
    figure(4); sgtitle('Transformed eigenvectors $u$', 'Interpreter', 'Latex');
    % RETRIEVE EMBEDDING --------------------------------------------------
    [uData, vData] = get_embedding(results, K, R);
    % RUN KMEANS ON EMBEDDING ---------------------------------------------
    figure(5)
    subplot(1, 2, 1)
    clustersu = cluster(uData, K, num_clusters_u, ...
        'Segmentation obtained using eigenfunctions $\{u_i\}$');
    subplot(1, 2, 2)
    clustersv = cluster(vData, K, num_clusters_v, ...
        'Segmentation obtained using eigenfunctions $\{v_i\}$');
end
% Plots the final segmentation using kmeans on the embedding
function clusters = cluster(data, K, num_clusters, s)
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
% Plots eigenfunction v_i
function plot_v(i, results, model)
    v = results.Eigenvectors;
    pdeplot(model,'XYData',v(:,i));
    s = strcat('$v_', num2str(i), '(x_1, x_2)$');
    title(s,'Interpreter', 'Latex', 'Fontsize', 14)
    xlabel('$x_1$', 'Interpreter', 'Latex', ...
        'Interpreter', 'Latex', 'Fontsize', 14)
    ylabel('$x_2$', 'Interpreter', 'Latex', ...
        'Interpreter', 'Latex', 'Fontsize', 14)
end
% Plots eigenfunction u_i
function plot_u(i, results, r)
    u = getu(i, results, r); 
    imagesc(u)
    colorbar
    s = strcat('Plot of $u_', num2str(i), '$ mapped to image pixels');
    title(s,'Interpreter', 'Latex', 'Interpreter', 'Latex', ...
        'Fontsize', 14)
    xlabel('pixel vertical coordinate', 'Interpreter', 'Latex', ...
        'Fontsize', 14)
    ylabel('pixel horizontal coordinate', 'Interpreter', 'Latex', ...
        'Fontsize', 14)
end
% Plots model domain. 
function plot_domain(model)
    pdegplot(model,'EdgeLabel','on');
    xlim([-0.5,1.5])
    ylim([-0.5,1.5])
    title('Plot of domain $\Omega$', 'Interpreter', 'Latex', 'Fontsize', 14)
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