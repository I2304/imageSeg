clear all; close all; clc; 

% Example: 
% evaluate_segmentation(1, 1, 1, 60, 2, false, 0.25);  


% BALANCED CASE
accuracies = zeros(9, 5); 
params = [...
    .5, 1, .5; ...
    2, 2, 0; ...
    1.5, 2, .5; ...
    1, 2, 0; ...
    .5, 1.5, .5; ...
    1, 2, .5; ...
    1, 1, 1; ...
    2, 1, 0; ...
    1.5, 1, .5; ...
];
for m = 1:9
    row = params(m, :); 
    [l, QC, QT] = evaluate_segmentation(row(1), row(2), row(3), 60, 4, ...
        true, 0.03); 
    accuracies(m, 1:3) = row; 
    accuracies(m, 4) = QT; 
    accuracies(m, 5) = QC; 
end

accuracies

% Takes in: 
%  P: the value of p to be used in the normalization
%  Q: the value of q to be used in the normalization
%  R: the value of r to be used in the normalization 
%  maxL: the maximum magnitude eigenvalue (e.g., 40) to be solved for
%  index: the id of the image (clean images: ids 1-4)
%  noise: true if noise wanted 
%  intensity: intensity of noise when noise == true (0~1)
function [l, QC, QT] = evaluate_segmentation(P, Q, R, maxL, id, noise, intensity)
    global M   
    global rho_matrix
    global q 
    global pr 
    
    if Q==P+R
        dest = ['./validation_results/balanced/', ...
            nextname('./validation_results/balanced/figure_set','_1','')];
    elseif Q < P+R
        dest = ['./validation_results/smallerq/', ...
            nextname('./validation_results/smallerq/figure_set','_1','')];
    else 
        dest = ['./validation_results/biggerq/', ...
            nextname('./validation_results/biggerq/figure_set','_1','')];
    end
    mkdir(dest);
    epsilon = 10^(-3);
    % LOAD & PRE-PROCESS IMAGE --------------------------------------------
    [img, truth, num_clusters] = test_img(id, noise, intensity);
    K = num_clusters-1; 
    figure(1)
    a1 = subplot(2, 2, 1);
    imagesc(img)
    colormap(a1, gray);
    colorbar
    xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 11)
    ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 11)
    title ('Original Image', 'Interpreter', 'Latex', 'Fontsize', 11)
    a2 = subplot(2, 2, 2);
    imagesc(truth)
    colormap(a2, jet); 
    colorbar
    xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 11)
    ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 11)
    title ('Ground Truth', 'Interpreter', 'Latex', 'Fontsize', 11)
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
    figure(2); 
    s = strcat('Eigenfunctions for ($p$, $q$, $r$) = (', ...
        num2str(P), ',', num2str(Q),',', num2str(R), ')');
    [~,h] = suplabel(s, 't');
    set(h, 'Interpreter', 'Latex', 'Fontsize', 12);
    temp=[dest,filesep,'eigenfunctions.png'];
    if id == 1
        set(gcf,'Position',[100 100 800 300])
    elseif id == 2
         set(gcf,'Position',[100 100 1200 400])
    elseif id == 3
        set(gcf,'Position',[100 100 700 500])
    elseif id == 4
        set(gcf,'Position',[100 100 1200 500])
    end
    saveas(gca,temp);
    % RETRIEVE EMBEDDING --------------------------------------------------
    [uData, ~] = get_embedding(results, K, R);
    
    % RUN KMEANS ON EMBEDDING ---------------------------------------------
    figure(1)
    a3 = subplot(2, 2, 3);
    cluster(uData, truth, K, num_clusters, a3, 10, 0, 1);
    s = strcat('Comparison for ($p$, $q$, $r$) = (', ...
        num2str(P), ',', num2str(Q),',', num2str(R), ')');
    [~,h] = suplabel(s, 't');
    set(h, 'Interpreter', 'Latex', 'Fontsize', 12);
    temp=[dest,filesep,'comparisons.png'];
    h = [a1, a2, a3]; 
    pos = get(h,'Position');
    new = mean(cellfun(@(v)v(1),pos(1:2)));
    set(h(3),'Position',[new,pos{end}(2:end)])
    set(gcf,'Position',[100 100 600 500])
    saveas(gca,temp);
    
    figure(3)
    qualities_t = 1:3;
    qualities_c = 1:3; 
    for i = 1:30
        if i <= 9
            a3 = subplot(3, 3, i);
        end
        [q_t, q_c] = cluster(uData, truth, K, num_clusters, a3, 1, i, 9);
        qualities_t(i) = q_t;
        qualities_c(i) = q_c;  
    end
    s = strcat('Examples for ($p$, $q$, $r$) = (', ...
        num2str(P), ',', num2str(Q),',', num2str(R), ')');
    [~,h] = suplabel(s, 't');
    set(h, 'Interpreter', 'Latex', 'Fontsize', 12);
    s = strcat('TA averaged over 30 reps: ', {' '}, num2str(mean(qualities_t)));
    disp(s); 
    s = strcat('CA averaged over 30 reps: ', {' '}, num2str(mean(qualities_c)));
    disp(s)
    temp=[dest,filesep,'sample_segmentations.png'];
    set(gcf,'Position',[100 100 700 500])
    saveas(gca,temp);
    QT = mean(qualities_t); 
    QC = mean(qualities_c); 
end
% Plots the final segmentation using kmeans on the embedding
function [quality_t, quality_c] = cluster(data, truth, K, ...
    num_clusters, a3, reps, i, display)
    X = reshape(data, [(length(data))^2, K]);
    clusters = reshape(kmeans(X, num_clusters, 'Replicates', reps), ...
        [(length(data)), (length(data))]); 
    [quality_t, quality_c] = compute_quality(clusters, truth);
    if i <= display
        imagesc(clusters)
        colormap(a3, jet); 
        colorbar
        s = strcat('TA =', {' '}, num2str(round(quality_t, 2)), ...
            '; CA =', {' '}, num2str(round(quality_c, 2)));
        title(s, 'Interpreter', 'Latex', 'Fontsize', 11)
        xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 11)
        ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 11)
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

% Plots eigenfunction u_i
function plot_varphi(i, results, r)
    u = getu(i, results, r); 
    imagesc(u)
    colorbar
    s = strcat('$\varphi_', num2str(i), '$');
    title(s,'Interpreter', 'Latex', 'Interpreter', 'Latex', ...
        'Fontsize', 11)
    xlabel('pixel $i$', 'Interpreter', 'Latex', ...
        'Fontsize', 11)
    ylabel('pixel $j$', 'Interpreter', 'Latex', ...
        'Fontsize', 11)
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

function [quality_t, quality_c] = compute_quality(clusters, truth)
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
    % overlal quality
    quality_t = sum(sum(truth==cpy))/(length(cpy))^2;
    % quality of clusters only
    quality_c = (sum(sum((truth==cpy).*(truth~=1))))/(sum((sum(truth~=1)))); 
end

% Compute cost matrix
function cost_matrix = make_cost_matrix(confusion_matrix)
    s = max(confusion_matrix); 
    cost_matrix = -confusion_matrix + s; 
end