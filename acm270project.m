
clear all; close all; clc; 

%% Main execution block 

% Example segmentations. First, use contrastImage_template.m to properly
% adjust the contrast of any image you need to segment (in order to 
% avoid singularity errors).

% l = segment_image('./us_two_inc/density_contrasted.png', 3/2, 1, 1/2, 2, 3);
l = segment_image('./us_usnccm/density_contrasted.png', 3/2, 1, 1/2, 2, 3);

%% Image segmentation function 
% Takes in: 
%  path: the (relative) path to an image 
%  P: the value of p to be used in the normalization
%  Q: the value of q to be used in the normalization
%  R: the value of r to be used in the normalization 
%  minI: minimum eigenfunction (e.g., 2) to be plotted
%  maxI: maximum eigenfunction (e.g., 5) to be plotted
% Plots the specified range of eigenfunctions v_{minI}, ..., v_{maxI} to 
% be plotted, and returns l (the list of eigenvalues identified). 
function l = segment_image(path, P, Q, R, minI, maxI)
    global M   
    global rho_matrix
    global q 
    global pr 

    % LOAD & PRE-PROCESS IMAGE --------------------------------------------
    img = imread(path);
    img = cast(img, 'double')/255+.01;
    figure()
    imshow(img)
    title ('Original Image', 'Interpreter', 'Latex', 'Fontsize', 14)

    % SET GLOBAL VARIABLES ------------------------------------------------
    rho_matrix = img;
    M = length(img); 
    q = Q; 
    pr = P + R;

    % SPECIFY A SMALLE TOLERANCE EPSILON ----------------------------------
    epsilon = 10^(-3);  


    % CREAT EVP MODEL -----------------------------------------------------
    model = construct_laplacian_model();
    % Plot the domain
    figure()
    plot_domain(model)
    % Calculate eigenvalues
    results = solve_evp(model,[-epsilon, 10],epsilon);
    l = results.Eigenvalues;

    % PLOTS ---------------------------------------------------------------
    for i = minI:maxI
        % Plot v_i
        figure()
        plot_v(i, results, model)
        % Plot u_i
        figure()
        plot_u(i, results, R)
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
    u = reshape(v,size(xq))./((rho_matrix).^(r));
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
    model.SolverOptions.AbsoluteTolerance = 5.0000e-09;
    model.SolverOptions.RelativeTolerance = 5.0000e-06;
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