function [TA, CA] = get_accuracy(clusters, truth)
    confusion_matrix=confusionmat(...
        reshape(truth, [size(truth, 1)^2, 1]),...
        reshape(clusters, [size(clusters, 1)^2, 1]));
    s = max(confusion_matrix); 
    cost_matrix = -confusion_matrix + s; 
    [assignment, ~] = munkres(cost_matrix);
    cpy = clusters; 
    for i = 1:length(assignment)
        for j = 1:length(assignment)
            if assignment(i, j) == 1
             cpy(clusters == j) = i; 
            end
        end
    end
    % overall quality
    TA = sum(sum(truth==cpy))/(length(cpy))^2;
    % quality of clusters only
    CA = (sum(sum((truth==cpy).*(truth~=1))))/(sum((sum(truth~=1)))); 
end