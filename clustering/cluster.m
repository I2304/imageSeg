% Plots the final segmentation using kmeans on the embedding
function[clusters] = cluster(data, k, K, reps)
    X = reshape(data, [(length(data))^2, k]);
    clusters = reshape(kmeans(X, K, 'Replicates', reps), ...
        [(length(data)), (length(data))]);
end