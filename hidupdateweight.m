
function [params] = hidupdateweight(params, new_class, data)

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
%     if(params.w(new_class, :)*x' > 0)
%         % update it for misclassification
%         learnrate = (1- params.w(new_class, :)*x')/sum(x.^2);
%         params.w(new_class, :) = params.w(new_class, :) - min(alpha, learnrate)*x;
%     end
    
%     if(params.w(new_class, :)*x' > 0)
%         % update it for misclassification
%         learnrate = (1- params.w(new_class, :)*x')/sum(x.^2);
%         params.w(new_class, :) = params.w(new_class, :) - min(alpha, learnrate)*x;
%     end
    
    
    
%     % delete all corresponding weight
%     scores = zeros(1, params.num_classes);
%     for cid = 1: params.num_classes
%         scores(cid) = params.w(cid, :)*x';
%     end
%     [val, idx]  = max(scores);
%     loss = scores(idx) + 1 - scores(new_class);
%     if idx~=new_class && loss > 0
%        % update the model here
%        
%        learnrate =  loss/sum(x.^2);
%        params.w(new_class, :) = params.w(new_class, :) + min(alpha, learnrate)*x;
%        params.w(idx, :) = params.w(new_class, :) - min(alpha, learnrate)*x;
%     end

   params.w(new_class, :) = params.w(new_class, :) - alpha*params.w(new_class, :);
end