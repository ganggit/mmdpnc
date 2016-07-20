% Copyright (C) 2014 Gang Chen, gangchen@buffalo.edu
% distributable under GPL, see README.txt

[Y,z,mu,ss,p] = drawGmm(200);
subplot(1,2,1);
title('generative clusters');
scatterMixture(Y,z);
params = dpmm_mmc(Y,100);
subplot(1,2,2);
title('dpmm_mmc clustering');
scatterMixture(Y,params(end).classes);
