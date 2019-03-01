function params = unhidupdateweight_mmc(params,new_class, data, flag, idxlist)
% score = zeros(1, params.num_classes);
% classification

learnrate = 0.01;
if nargin <4
    flag = false;
end


if nargin ==5
    label = params.classes(idxlist);
    prior = histc(label,1:params.num_classes);%
    prior = 1 - prior./sum(prior); % adust weight for each elements
else
    prior  = ones(1, params.num_classes)/params.num_classes;
end



if flag
    data = [ones(size(data,1),1), data];
end
maxval = -100000;
score = 0;
clsid =1;

for didx = 1: size(data,1)
    x = data(didx,:);
    for i = 1: params.num_classes

        % classify it
        val = params.w(i,:)*x';
        % score(i) = params.w(i,:)*[1 x]';
        if i==new_class
            score = val;
        else
            if(val>maxval)
                maxval = val;
                clsid = i;
            end
        end

    end
    % % find the maximum one
    % [val, clsid] = max(score);
    loss  = maxval +1 - score;
    if(loss>0)
        % update the model;
        params.w(new_class,:) = params.w(new_class,:) + min(learnrate, 0.5*loss/sum(x.^2))*x;
        params.w(clsid,:) = params.w(clsid,:) - min(learnrate, 0.5*loss/sum(x.^2))*x;

    end
    dim = size(params.w, 2);
    % params.w = params.w./repmat(sqrt(sum(params.w.^2, 2))+0.001, [1 dim]);

    % if(clsid~=new_class)
    %     % update the model;
    %     params.w(new_class,:) = params.w(new_class,:) + min(learnrate, prior(new_class)*loss/sum(x.^2))*x;
    %     params.w(clsid,:) = params.w(clsid,:) - min(learnrate, prior(clsid)*loss/sum(x.^2))*x;
    %     % normalize it
    %     neww = params.w;
    %     dim = size(neww, 2);
    %     neww = neww./repmat(sqrt(sum(neww.^2, 2))+0.001, [1 dim]);
    %     params.w = neww;
    %     % params.counts(clsid) = params.counts(clsid) - size(x,1);
    %     
    % end
end

params.counts(new_class) = params.counts(new_class) + size(data,1);
