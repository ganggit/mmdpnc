% Copyright (C) 2007 Jacob Eisenstein: jacobe at mit dot edu
% distributable under GPL, see README.txt

function params = addNewClass_mmc(params, flag)
%function params = addNewClass(params)
%adds a new, empty class to the dpmm
if nargin <2
    flag = false;
end
    newclassidx = params.num_classes+1;
    params.num_classes = newclassidx;
    params.counts(newclassidx) = 0;
    params.sums(newclassidx,:) = params.kappa * params.initmean;
    % params.cholSSE(:,:,newclassidx) = chol(params.nu * params.initcov);
    %params.SSE(:,:,newclassidx) = params.nu * params.initcov;
    dim = size(params.initmean,2); % get the data dimention
    if flag 
        dim = dim +1;
    end
    % try to find an orthogonal direction
    % neww = zeros(1,dim);
    if(size(params.w, 1)>=params.num_classes)
        params.w = params.w(1:params.num_classes, :);
    else
        %% one way to try new w
        r = randn(1,dim);
        try
            for ii  =1: newclassidx-1
                %res = r - (r*params.w(ii,:)')*params.w(ii,:); %/sum(r.^2);
                %r = res;
                % res = r - params.w(ii,:)*params.w(ii,:)'*r;%/sum(params.w(ii,:).^2);
                res = r - params.w(ii,:).*(r*params.w(ii,:)')/sqrt(sum(r.^2));
                r = res;
            end
        catch
            stop =1;
        end
        
        %% another way to try w
        a =2; b = 0.9;
        beta = 0.05*eye(dim);
        mu0 = zeros(1, dim);
        % location = mu
        n = 2*a;
        % y ~ N(0,inv(a/b*beta));  
        sigma = 1/sqrt(a/b*beta(1)); % since beta is diagonal
        y   = randn(1,dim)*sigma;
        % u ~ chi-square(n)        
        u   = chi2rnd(2*a);
        % see multivariate student distribution on wiki
        mu   = y*sqrt(n/u)+params.initmean;
        %res = mu;
        
        % res = mean(params.w(1:newclassidx-1,:),1);
        % res = res./sum(res.^2); % add late
        % res = params.initmean/sqrt(sum((params.initmean).^2));
        
        %% try here
        % kappabar = params.counts + params.kappa;
        % res = params.sums(newclassidx,:)/ kappabar(newclassidx);
        
        neww = zeros(newclassidx, dim);
        neww(1:newclassidx-1, :) = params.w;
        neww(newclassidx, :) = res;
        
        % normalize it
        % neww = neww./repmat(sqrt(sum(neww.^2, 2))+0.001, [1 dim]);
        params.w = neww;
    end
end
