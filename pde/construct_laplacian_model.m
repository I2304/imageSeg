
function [model, results] = construct_laplacian_model(rho, p, q, r, M, lim)

    % Concstruct geometry and basic model
    lowerLeft  = [0 , 0 ];
    lowerRight = [1 , 0 ];
    upperRight = [1 , 1];
    upperLeft =  [0 , 1];
    S = [3,4 lowerLeft(1), lowerRight(1), upperRight(1), upperLeft(1), ...
             lowerLeft(2), lowerRight(2), upperRight(2), upperLeft(2)];
    gdm = S';
    ns = 'S';
    sf = 'S';
    g = decsg(gdm,ns,sf');
    model = createpde();
    geometryFromEdges(model,g);
    model.SolverOptions.AbsoluteTolerance = 5.0000e-025;
    model.SolverOptions.RelativeTolerance = 5.0000e-025;
    model.SolverOptions.ReportStatistics = 'on';
    
    % Homogeneous Neumann BCs and coefficients
    applyBoundaryCondition(model,'neumann','Edge',1:4,'g',0,'q',0);
    c_fun = @(location, state)ccoeffunction(location, state, rho, q, M);
    d_fun = @(location, state)dcoeffunction(location, state, rho, p, r, M);
    specifyCoefficients(model,'m',0,'d',d_fun,...
        'c',c_fun,'a',0,'f',0);
    
    % Solve EVP
    generateMesh(model,'Hmax',0.01)
    eps = 10^(-3);
    range = [-eps, lim]; 
    results = solvepdeeig(model,range);
end