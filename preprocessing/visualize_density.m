function density = visualize_density(density, figure_no, filename)
    figure(figure_no);
    imagesc(density)
    colormap(gray)
    c = colorbar;
    ylabel(c, 'Intensity', 'Interpreter', 'Latex', 'Fontsize', 14)
    title ('Original Image', 'Interpreter', 'Latex', 'Fontsize', 14)
    dest = ['./res_images/',filename];
    mkdir(dest);
    saveas(figure(figure_no),[pwd, '/res_images/', filename, '/original.fig']);
end