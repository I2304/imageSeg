% Computes the embedding of the image from the eigenfunctions
function uData = get_embedding(results, K, r, M, rho)
    uData = zeros(M, M, K);
    for k = 1:K
        [xq,yq] = meshgrid(linspace(0, 1, M));
        v = interpolateSolution(results,xq,yq,k);
        u = flip(reshape(v,size(xq)), 1).*((rho).^(r));
        uData(:, :, k) = u;
    end
end