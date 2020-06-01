
function [TA, CA] = run_segmentation(...
    type, ...
    spec, ...
    swap, ...
    p, ...
    q, ...
    r, ...
    lim, ...
    k, ...
    K, ...
    filename, ...
    truth...
)
    addpath('./preprocessing')
    addpath('./pde')
    addpath('./file_exchange')
    addpath('./testing')
    addpath('./clustering')
    addpath('./clustering/fuzme_matlab')
    
    % Synthetic image
    if exist('truth','var')
        synthetic = true; 
        rho = visualize_density(spec, 1, filename); 
    % Natural image
    else
        synthetic = false; 
        rho = preprocess_image(spec, swap, 1, filename);
    end
    
    % Segment
    M = length(rho); 
    [~, results] = construct_laplacian_model(rho, p, q, r, M, lim);
    plot_eigenfunctions(k, 2, results, p, q, r, M, rho, filename)
    u_data = get_embedding(results, k, r, M, rho);
    if strcmp(type, 'kmeans')
        clusters = cluster(u_data, k, K, 5);
        if synthetic
            [TA, CA] = get_accuracy(clusters, truth);
            plot_clusters(clusters, 3, filename, p, q, r, k, K, false, TA, CA);
        else
            TA = -1; CA = -1; 
            plot_clusters(clusters, 3, filename, p, q, r, k, K);
        end
    elseif strcmp(type, 'soft')
        TA = -1; CA = -1; 
        figure(3)
        fuzzy(u_data, 3, p, q, r, k, K, filename); 
    end
end