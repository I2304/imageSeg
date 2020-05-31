function [img] = preprocess_image(path, swap, figure_no, filename)
    img = mat2gray(imread(path));
    % Force image to be square
    img = img(1:min(size(img, 1), size(img, 2)), ...
        1:min(size(img, 1), size(img, 2)));
    if swap == true
        img = 1-(cast(img, 'double'));
    end
    % Make a small correction to zero terms (to avoid numerical issues)
    img(img == 0) = .01;
    % Plot
    figure(figure_no);
    imagesc(img)
    colormap(gray)
    c = colorbar;
    ylabel(c, 'Intensity', 'Interpreter', 'Latex', 'Fontsize', 14)
    title ('Original Image', 'Interpreter', 'Latex', 'Fontsize', 14)
    dest = ['./res_images/',filename];
    mkdir(dest);
    saveas(figure(figure_no),[pwd, '/res_images/', filename, '/original.fig']);
end