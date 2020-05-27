
% This function defines all test cases. It takes in an index i and 
% returns the corresponding synthetic test image rho_dense and 
% groun_truth rho_truth. 

% Returns test case i without noise. 
function [rho_dense, rho_truth, num_clusters] = test_img(i)
    if i == 1
        rho_dense = .01*ones(1000); % Two squares. 
            rho_dense(200:400, 200:400) = 1; 
            rho_dense(500:600, 500:600) = 1; 
        rho_truth = 1*ones(1000); 
            rho_truth(200:400, 200:400) = 2; 
            rho_truth(500:600, 500:600) = 3; 
        num_clusters = 3; 
    elseif i == 2
        rho_dense = .01*ones(1000); % Two squares, nested 
        rho_truth = 1*ones(1000); 
            rho_dense(200:800, 200:800) = 1; 
            rho_truth(200:800, 200:800) = 2; 
            rho_dense(300:700, 300:700) = .01;
            rho_truth(300:700, 300:700) = 1;
            rho_dense(400:600, 400:600) = 1;
            rho_truth(400:600, 400:600) = 3;
            rho_dense(600:800, 100:150) = 1;
            rho_truth(600:800, 100:150) = 4;
            rho_dense(650:750, 110:140) = .01;
            rho_truth(650:750, 110:140) = 1;
        num_clusters = 4; 
    elseif i == 3
        rho_dense = .01*ones(1000); % Multiple objects, nested
        rho_truth = ones(1000); 
            for i = 1:1000
                for j = 1:1000
                    if i <= 400 && j <= 400 && i >= 100 && j >= 100
                        rho_dense(i, j) = 1; 
                        rho_truth(i, j) = 2; 
                    end
                    if (i-700)^2+(j-700)^2 <= 8000
                        rho_dense(i, j) = 1; 
                        rho_truth(i, j) = 3; 
                    end
                    if (i-600)^2+(j-600)^2 <= 800
                        rho_dense(i, j) = 1; 
                        rho_truth(i, j) = 4; 
                    end
                    if (i-800)^2+(j-600)^2 <= 400
                        rho_dense(i, j) = 1; 
                        rho_truth(i, j) = 5; 
                    end
                    if (i-900)^2+(j-700)^2 <= 400
                        rho_dense(i, j) = 1; 
                        rho_truth(i, j) = 6; 
                    end
                    if i <= 300 && j <= 800 && i >= 200 && j >= 500 && i+500 <= j
                        rho_dense(i, j) = 1;
                        rho_truth(i, j) = 7;
                    end
                end
            end
            rho_dense(150:350, 150:350) = .01; 
            rho_truth(150:350, 150:350) = 1; 
            rho_dense(150:400, 200:230) = 1; 
            rho_truth(150:400, 200:230) = 2; 
            rho_dense(275:325, 275:325) = 1; 
            rho_truth(275:325, 275:325) = 8; 
            rho_dense(200:225, 275:300) = 1; 
            rho_truth(200:225, 275:300) = 9; 
            rho_dense(800:900, 100:200) = 1; 
            rho_truth(800:900, 100:200) = 10;
            num_clusters = 10; 
    end
    if i == 4
        rho_dense = .01*ones(1000);
        rho_truth = 1*ones(1000); 
        for i = 1:1000
            for j = 1:1000
                if (i-500)^2+(j-500)^2 <= 8000
                    rho_dense(i, j) = 1; 
                    rho_truth(i, j) = 2; 
                end
                if (i-500)^2+(j-500)^2 <= 3000
                    rho_dense(i, j) = .01; 
                    rho_truth(i, j) = 1; 
                end
                if (i-630)^2+(j-630)^2 <= 12000
                    rho_dense(i, j) = 1; 
                    rho_truth(i, j) = 2; 
                end
                if (i-370)^2+(j-370)^2 <= 12000
                    rho_dense(i, j) = 1; 
                    rho_truth(i, j) = 2; 
                end
                if (i-370)^2+(j-370)^2 <= 12000
                    rho_dense(i, j) = 1; 
                    rho_truth(i, j) = 2; 
                end
                if (i-200)^2/8+(j-800)^2/1 <= 2000
                    rho_dense(i, j) = 1; 
                    rho_truth(i, j) = 3; 
                end
                if (i-900)^2 + (j-750)^2/8 <= 10000
                    rho_dense(i, j) = 1; 
                    rho_truth(i, j) = 4; 
                end
                if i <= 600 && j <= 300 && i >= 500 && j >= 200 && ...
                        (i-500) >= (j-200) || ...
                    i <= 700 && j <= 300 && i >= 600 && j >= 200 && ...
                        (i-600) >= (j-200) || ...
                    i <= 800 && j <= 300 && i >= 700 && j >= 200 && ...
                        (i-700) >= (j-200) 
                    rho_dense(i, j) = 1;
                    rho_truth(i, j) = 5;
                end
                if i <= 900 && i >= 850 && j >= 300 && j <= 400
                    rho_dense(i, j) = 1;
                    rho_truth(i, j) = 6;
                end
            end
        end
        rho_dense(50:150, 50:150) = 1;
        rho_truth(50:150, 50:150) = 7;
        rho_dense(175:275, 175:275) = 1;
        rho_truth(175:275, 175:275) = 8;
        num_clusters = 8; 
    end
end
