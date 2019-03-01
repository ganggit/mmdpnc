function [mF, precision, recall, rand, jaccard] = F_measure(label, result)

N = numel(label);

cls_num = unique(label);
res_num = unique(result);
Ppairs = zeros(N*(N-1),1); % standard or ground truth
idx = 0;
for i =1: N-1
    for j = i+1: N
        
       if(label(i) ==label(j))
          idx = idx + 1; 
          Ppairs(idx) = sub2ind([N,N], i,j);
       end
    
    end
end
Ppairs(idx+1:end) = []; % delete empty pair
Qpairs = zeros(N*(N-1),1); % our algorithm or prediction
idx = 0;
for i =1: N-1
    for j = i+1: N
        
       if(result(i) ==result(j))
          idx = idx + 1; 
          Qpairs(idx) = sub2ind([N,N], i,j);
       end
    
    end
end
Qpairs(idx+1:end) = [];

Allpairs = zeros(N*(N-1),1);
idx = 0;
for i =1: N-1
    for j = i+1: N      
        idx = idx + 1; 
        Allpairs(idx) = sub2ind([N,N], i,j);
    end
end
Allpairs(idx+1: end) = []; % delete empty pair

% intersection
a = intersect(Ppairs, Qpairs);
b = setdiff(Ppairs, Qpairs);
c = setdiff(Qpairs, Ppairs);
d = setdiff(Allpairs, Ppairs);
d = setdiff(d, Qpairs);
mF = 2*numel(a)/(2*numel(a)+numel(b)+numel(c));

precision = numel(a)/(numel(a)+numel(c));
recall = numel(a)/(numel(a)+numel(b));
rand = (numel(a) + numel(d))/(numel(a)+numel(b) + numel(c) + numel(d));
jaccard = (numel(a))/(numel(a)+numel(b) + numel(c));


