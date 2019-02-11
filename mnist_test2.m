
addpath('/home/gangchen/Downloads/data/UCI');


%fname = 'iris.data'; 
fname = 'wine.data'; 

% fname = 'R15.txt';%'jain.txt'; % spiral.txt

% fname = 'glass.data.txt';
% fname = 'wdbc.data.txt';
fname = 'digits.mat';
%fname = 'Aggregation.txt';
%fname = 'spiral.txt';
% fname = 'news20.mat';
% fdir = '/home/gangchen/Downloads/object/hclassification/multisvm/DataSet/20news-bydate/matlab/';
fdir = './';
if strcmp(fname(end-2:end),'mat')
    load([fdir, fname]);
    clear testdata;
    clear testlabel;
    feat = double(traindata)/255;
    clsid = trainlabel;
    clear traindata;
    clear trainlabel;
else
    [feat, clsid] = readspiral(fname,  './');
end
% [feat, clsid] = readspiral(fname,  '/home/gangchen/Downloads/data/UCI/'); % 

% reiterating 10 times
numiters = 10;
[numdata, dim] = size(feat);
numsubdata = min(floor(numdata/numiters),2000);

dpm = zeros(numiters, 3);
dpmmc = zeros(numiters, 3);
for i = 1: numiters
disp('dpm clustering');
ids = randperm(numdata);    
numsub = i*numsubdata;
tmp = feat(ids(1:numsub),:);
m= mean(tmp, 1);
mO = tmp - repmat(m, numsub,1);
CV=mO'*mO;
[V D]=eig(CV);
D = diag(D);
[val,idlist]  = sort(-D);
cs = cumsum(-val/sum(-val));
idx = find(cs > 0.95, 1);
if(idx>100)
    idx = 100;
end
newV=V(:,idlist(1:idx)); 
data=tmp*newV;
% data = tmp;

% subplot(1,2,1);
% scatterMixture(feat,clsid);
% disp('show dataset');
%[params, tlast] = dpmm(data, 50, numsub/50, 0.5, idx+2);
% subplot(1,2,2);
% title('dpmm clustering');
% scatterMixture(feat,params(end).classes);
% val = nmi(clsid,params(end).classes);

%% using the dpm for clustering
% [mF] = F_measure(clsid(ids(1:numsub)),params(end).classes);
% [conmatrix] = contable(clsid(ids(1:numsub)),params(end).classes);
% % [purity, entropy,mF] = eval_clustering(conmatrix);
% [v,hc,hk,h_ck,h_kc] = calculate_v_measure(conmatrix);




disp('dp_mmc clustering');
[params, tlast] = dpmm_mmc(data,50, 4);
[mF] = F_measure(clsid(ids(1:numsub)),params(end).classes);
[conmatrix] = contable(clsid(ids(1:numsub)),params(end).classes);
% [purity, entropy, mF] = eval_clustering(conmatrix);
[v,hc,hk,h_ck,h_kc] = calculate_v_measure(conmatrix);
dpmmc(i,1) = mF; dpmmc(i,2) = v; dpmmc(i,3) = tlast;

end

save('digitseval.mat', 'dpm', 'dpmmc');


% show the MNIST results
load digitseval.mat;

x = (1:10)*200;
y1 = dpm(:,3);
y2 = dpmmc(:,3);
%plot(x, y1, '-rs', x, y2, 'dg')
%plot(x, y1, '-rs', x, y2, '-db'); % old display
plot(x, y1, '-bs', x, y2, '-rd', 'LineWidth',1.1);
%hleg1 = legend('DPM','MMDPM');
hleg1 = legend('DPM','NMMC');
xlabel('the number of training samples', 'FontSize',14);
ylabel('the time of training (seconds)', 'FontSize',14);
title('time changes with the number of training samples', 'FontSize',14);

axis([200 2000 0 2500]);


% show the news20 results
load news20_varall.mat;

x = 50:50:350;
y1 = [dpm(:,3); 440]*10;
y2 = [dpmmc(:,3); 135]*10;
%plot(x, y1, '-rs', x, y2, 'dg')
%plot(x, y1, '-rs', x, y2, '-db'); % old display
plot(x, y1, '-bs', x, y2, '-rd', 'LineWidth',1.1);
hleg1 = legend('DPM','NMMC');
xlabel('the data dimensionality', 'FontSize',14);
ylabel('the time of training (seconds)', 'FontSize',14);
title('time changes with the number of dimensions', 'FontSize',14);

axis([0 400 0 6000]);



