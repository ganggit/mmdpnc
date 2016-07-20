
function [params] = hidupdateweight_mmc(params, new_class, data)

alpha = 0.01;
if nargin < 4
    flag = false;
end
%must add iteratively because of the cholesky update function
for i = 1:size(data,1)
    params.counts(new_class) = params.counts(new_class) - 1;
    params.sums(new_class,:) = params.sums(new_class,:) - data(i,:);
    if flag
        x = [1 data(i,:)];
    else
        x = data(i,:);
    end

   params.w(new_class, :) = params.w(new_class, :) - alpha*params.w(new_class, :);
end