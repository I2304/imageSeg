%file for generating test images
clear;
for id = 1:4
    img1 = test_img(id, false, 0.25);
    plot_img(img1, ["Synthetic", "Image", string(id)])

    img1 = test_img(id, true, 0.03);
    plot_img(img1, ["Synthetic", "Image", string(id), "with", "Noise"])
end

function [] = plot_img(img, plot_title)
    figure()
    imagesc(img)
    c = colorbar; 
    ylabel(c, '[0, 1] Intensity', 'Interpreter', 'Latex', 'Fontsize', 11)
    xlabel('pixel $i$', 'Interpreter', 'Latex', 'Fontsize', 11)
    ylabel('pixel $j$', 'Interpreter', 'Latex', 'Fontsize', 11)
    t = join(plot_title, ' ');
    title (t, 'Interpreter', 'Latex', 'Fontsize', 11)
    file_t = join(plot_title, '_');
    saveas(gcf, join(["./test_images/", file_t, ".png"], ''))
    
end
