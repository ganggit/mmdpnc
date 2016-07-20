function params = removeClasses_mmc(params)
if (~isfield(params,'num_fixed') || params.num_fixed == inf)
idxs = find(params(end).counts == 0);
for ctr = idxs
    %reduce all state numbers that are greater than ctr
    params.classes = params.classes - (params.classes >= ctr);
    idxs2 = [1:(ctr-1) (ctr+1):params.num_classes];
    params.counts = params.counts(idxs2);
    params.sums = params.sums(idxs2,:);
    params.num_classes = params.num_classes-1;
    
    try
        % add by Gang Chen
        params.w = params.w(idxs2,:);
    catch
        stop = 1;
    end
end
end
end