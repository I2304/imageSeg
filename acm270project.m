clear all; close all; clc; 

%% Main execution block 

% Example segmentations
% WORKS BEST WITH num_clusters = K+1 

% All test cases here are for the normalized laplacian (3/2, 2, 1/2) = (P,
% Q, R). These parameters can be tuned 

% [l, uData, vData] = get_segmentation('./test_cases/density_2_dots/original.png', ...
%     3/2, 2, 1/2, 2, 70, 3, 3, true);

% [l, uData, vData] = get_segmentation('./test_cases/brain/original.png', ...
%     3/2, 2, 1/2, 9, 70, 10, 10, false);

% [l, uData, vData] = get_segmentation('./test_cases/pattern_12_dots/original.png', ...
%     3/2, 2, 1/2, 11, 80, 12, 12, true);

% [l, uData, vData] = get_segmentation('./test_cases/usnccm_7_letters/original.png', ...
%     3/2, 2, 1/2, 6, 70, 7, 7, true);

% [l, uData, vData] = get_segmentation('./test_cases/flower/original.jpg', ...
%     3/2, 1, 1/2, 4, 80, 5, 5, false);


q = 2; r = 1;
p_lst = [0 0.5 1 1.5 2 2.5 10 20];
[l, uData, vData] = get_segmentation('./test_cases/flower/original.jpg', ...
p_lst, q, r, 8, 40, 10, 10, false);

% p = 0.5; r = 0.5;
% q_lst = [0 0.5 1 1.5 2 2.5 10 20];
% [l, uData, vData] = get_segmentation('./test_cases/complicatedflower/original.png', ...
% p, q_lst, r, 8, 40, 10, 10, false);


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
function [l, uData, vData] = get_segmentation(path, P_lst, Q_lst, R_lst, K, maxL, ...
    num_clusters_u, num_clusters_v, swap)
    dest = ['./results/' datestr(datetime('now'))];
    mkdir(dest);
    num = 1;
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
    imshow(img)
    title ('Original Image', 'Interpreter', 'Latex', 'Fontsize', 14)
    temp=[dest,filesep,'Original image ', num2str(num),'.png'];
    saveas(gca,temp);
    num = num + 1;
    % SET GLOBAL VARIABLES ------------------------------------------------
    rho_matrix = img;
    M = length(img); 
    for P = P_lst
        for Q = Q_lst
            for R = R_lst
                q = Q; 
                pr = P + R;
                % CREAT EVP MODEL -----------------------------------------------------
                model = construct_laplacian_model();
                % Plot the domain
                figure(2)
                plot_domain(model)
                temp=[dest,filesep,'Domain: ', num2str(P),'_',num2str(Q),'_',num2str(R),'.png'];
                saveas(gca,temp);
                num = num + 1;
                % Calculate eigenvalues
                results = solve_evp(model,[-epsilon, maxL],epsilon);
                l = results.Eigenvalues;
                % EIGENFUNCTION PLOTS -------------------------------------------------
                [a, ~] = numSubplots(K);
                figure(3)
                for i = 1:K
                    % Plot v_i
                    subplot(a(1), a(2), i)
                    plot_v(i, results, model)
                end
                temp=[dest,filesep,'Eigenvector v: ',num2str(P),'_',num2str(Q),'_',num2str(R),'.png'];
                saveas(gca,temp);
                num = num + 1;
                
                figure(4)
                for i = 1:K
                    % Plot u_i
                    subplot(a(1), a(2), i)
                    plot_u(i, results, R)
                end
                temp=[dest,filesep,'Eigenvector u: ', num2str(P),'_',num2str(Q),'_',num2str(R),'.png'];    
                saveas(gca,temp);
                num = num + 1;
                %figure(3); %sgtitle('Transformed eigenvectors $v$', 'Interpreter', 'Latex');
                %figure(4); %sgtitle('Transformed eigenvectors $u$', 'Interpreter', 'Latex');
                % RETRIEVE EMBEDDING --------------------------------------------------
                [uData, vData] = get_embedding(results, K, R);
                % RUN KMEANS ON EMBEDDING ---------------------------------------------
                figure(5)
                cluster(uData, K, num_clusters_u, ...
                    ['Segmentation using $\{u_i\}$ : p = ' num2str(P) ' , q = ' num2str(Q) ' , r = ' num2str(R)])
                temp=[dest,filesep,'Segmentation using u: ', num2str(P),'_',num2str(Q),'_',num2str(R),'.png'];
                saveas(gca,temp);
                num = num + 1;
                figure(6)
                cluster(vData, K, num_clusters_v, ...
                    ['Segmentation using $\{v_i\}$ : p = ' num2str(P) ' , q = ' num2str(Q) ' , r = ' num2str(R)])
                temp=[dest,filesep,'Segmentation using v: ', num2str(P),'_',num2str(Q),'_',num2str(R),'.png'];
                saveas(gca,temp);
                num = num + 1;
            end
        end
    end
end
% Plots the final segmentation using kmeans on the embedding
function [classU, classV] = cluster(data, K, num_clusters, s)
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
    [M,N] = size(v);
    if i<=N
        pdeplot(model,'XYData',v(:,i));
        s = strcat('$v_', num2str(i), '(x_1, x_2)$');
        title(s,'Interpreter', 'Latex', 'Fontsize', 14)
        xlabel('$x_1$', 'Interpreter', 'Latex', ...
            'Interpreter', 'Latex', 'Fontsize', 14)
        ylabel('$x_2$', 'Interpreter', 'Latex', ...
            'Interpreter', 'Latex', 'Fontsize', 14)
    end
end
% Plots eigenfunction u_i
function plot_u(i, results, r)
    u = getu(i, results, r); 
    imagesc(u)
    colorbar
    s = strcat('$u_', num2str(i), '$');
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