
%% DONT USE THIS FILE-- use acm270project.m instead. 

%% Import image 

og = imread('./us_usnccm/density.png');
img = mat2gray(og);

img = img(1:min(size(img, 1), size(img, 2)), 1:min(size(img, 1), size(img, 2)));
figure(1)
imshow(img)
global M
M = length(img); 

global rho_matrix
rho_matrix = img; 
global q
q = 5; 
global pr
pr = 1; 

%% Create Laplacian model
% Coordinates
lowerLeft  = [0 , 0 ];
lowerRight = [1 , 0 ];
upperRight = [1 , 1];
upperLeft =  [0 , 1];
% Geometry matrix
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
figure(2)
pdegplot(model,'EdgeLabel','on');
xlim([0,1.5])
ylim([0,1.5])

applyBoundaryCondition(model,'neumann','Edge',1:4,'g',0,'q',0);

specifyCoefficients(model,'m',0,'d',@dcoeffunction,'c',@ccoeffunction,'a',0,'f',0);
r = [-1,.1];

generateMesh(model,'Hmax',0.1)
results = solvepdeeig(model,r);
mesh = results.Mesh;

l = results.Eigenvalues;
u = results.Eigenvectors;

figure(4)
fig = pdeplot(model,'XYData',u(:,2));
[xq,yq] = meshgrid(linspace(0, 1, M));
uintrp = interpolateSolution(results,xq,yq,2);
uintrp = reshape(uintrp,size(xq))./rho_matrix;
figure(5)
imagesc(uintrp)
colorbar

function dmatrix = dcoeffunction(location,state)
global rho_matrix
global pr
global M
nr = numel(location.x);
dmatrix = ones(1,nr);
    for index = 1:nr
        x = location.x(index);
        y = location.y(index);
        i = ceil(M*(1-y));
        j = ceil(M*x); 
        if i == 0
            i = 1; 
        end
        if j == 0
            j = 1; 
        end
        dmatrix(index) = 1;
    end
end

function cmatrix = ccoeffunction(location,state)
global rho_matrix
global q
global M
nr = numel(location.x);
cmatrix = ones(1,nr);
    for index = 1:nr
        x = location.x(index);
        y = location.y(index);
        i = ceil(M*(1-y));
        j = ceil(M*x); 
        if i == 0
            i = 1; 
        end
        if j == 0
            j = 1; 
        end
        cmatrix(index) = (rho_matrix(i, j))^q;
    end
end