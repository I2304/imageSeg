
clear all; close all; clc; 

%% Load an image. 

og = imread('./us_two_inc/density.png');
img = mat2gray(og);
img(:, :) = 1;
img(200:300, 200:300) = .05; 
img(500:600, 500:600) = .05;

% For simplicity, it is assumed that the image is approximately square, 
% and we just truncate the longer end. 
img = img(1:min(size(img, 1), size(img, 2)), ...
    1:min(size(img, 1), size(img, 2))); 

figure(1)
imshow(img)
title ('Original Image', 'Interpreter', 'Latex', 'Fontsize', 14)

%% Define (p, q, r)

global M
M = length(img);
global rho_matrix
rho_matrix = img;
global q
q = 2; 
global pr
pr = 2; 
r = 1/2; 

%% Create square domain 

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
% Import g into model using geometryFromEdges.
model = createpde;
geometryFromEdges(model,g);
% Plot the domain
figure(2)
pdegplot(model,'EdgeLabel','on');
xlim([0,1.5])
ylim([0,1.5])
title('Plot of domain $\Omega$', 'Interpreter', 'Latex', 'Fontsize', 14)

%% Set up eigenvalue problem 

applyBoundaryCondition(model,'neumann','Edge',1:4,'g',0,'q',0);
specifyCoefficients(model,'m',0,'d',@dcoeffunction,...
    'c',@ccoeffunction,'a',0,'f',0);
range = [0,40];
generateMesh(model,'Hmax',0.05)
results = solvepdeeig(model,range);
l = results.Eigenvalues; % stores eigenvalues 

%% Plots

% Plot second eigenvector v
figure()
plotv(2, results, model)
% Plot second eigenvector u
figure()
plotu(2, results, r)

%% Plot helper functions

% Plots eigenfunction v_i
function plotv(i, results, model)
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
function plotu(i, results, r)
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

% Takes in the results of the eigenvalue solver and returns corresponding 
% eigenfunction u_i.
function u = getu(i, results, r)
    global M
    global rho_matrix
    [xq,yq] = meshgrid(linspace(0, 1, M));
    v = interpolateSolution(results,xq,yq,i);
    u = reshape(v,size(xq))./(rho_matrix.^(r));
end

%% PDE Coefficients 

% This function encodes the coefficient d, which in our formulation is
% \rho^(p+r). T
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