function cmatrix = ccoeffunction(location,state, rho, q, M)
    nr = numel(location.x);
    cmatrix = ones(1,nr);
    % Iterate over all nodes in location, and evaluate the value of
    % rho^(p+r) pointwise. Store the result in d_matrix.
    for index = 1:nr
        % Obtain position in location vector
        x = location.x(index);  % x location in mesh
        y = location.y(index);  % y location in mesh
        % Map (x, y) --> (i, j) pixel in image
        i = ceil(M*(1-y));      % i position in image
        j = ceil(M*x);          % j position in image
        % Correction for edge case
        if i == 0
            i = 1;
        end
        if j == 0
            j = 1;
        end
        % Store result
        cmatrix(index) = (rho(i,j))^(q);
    end
end