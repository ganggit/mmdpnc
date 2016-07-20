# mmdpnc
maximum margin dirichlet process mixtures for clustering, refer to https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11828/11763

# Basic idea: 
it is a discriminative model for nonparameteric clustering, which leverages dirchlet process and maximum margin clustering

#Gibbs sampling: 
1. it uses dirichlet process as the prior to generate the number of clusters
2. the likelihood is from maximum margin model
3. inference is done based on posterior probability

# Learning
it is based on maximum margin online learning to update component parameters

# Parameter Setting: 
Check the parameter C in dpmm_mmc.m, which balances the contribution from prior and likelihood.

#Demo
demodpmm_mmc.m

#Reference
Maximum Margin Dirichlet Process Mixtures for Clustering, G. Chen, H. Zhang and C. Xiong. AAAI Conference on Artificial Intelligence (AAAI 2016).
