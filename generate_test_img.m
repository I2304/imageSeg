%file for generating test images
img1 = test_img(1, false, 0.25);
plot_img(img1, ["Test", "Image", "1"])

img2 = test_img(2, false, 0.25);
plot_img(img2, ["Test", "Image", "2"])

img3 = test_img(3, false, 0.25);
plot_img(img3, ["Test", "Image", "3"])

img4 = test_img(4, false, 0.25);
plot_img(img4, ["Test", "Image", "4"])

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
    saveas(gcf, join([file_t, ".png"], ''))

end
