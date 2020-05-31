function plot_eigenfunction(i, results, r, M, rho)
    [xq,yq] = meshgrid(linspace(0, 1, M));
    v = interpolateSolution(results,xq,yq,i);
    u = flip(reshape(v,size(xq)), 1).*((rho).^(r));
    imagesc(u)
    colorbar; 
    s = strcat('$\varphi_', num2str(i));
    xlabel('$i$', 'Interpreter', 'Latex', 'Fontsize', 14)
    ylabel('$j$', 'Interpreter', 'Latex', 'Fontsize', 14)
    title(strcat(s, '$'),'Interpreter', 'Latex', 'Interpreter', 'Latex', ...
        'Fontsize', 14)
end