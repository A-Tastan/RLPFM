# RLPFM
Robust Spectral Clustering: A Locality Preserving Feature Mapping Based on M-estimation

Dimension reduction is a fundamental task in spectral clustering. In practical applications, the data may be corrupted by outliers and noise, which can obscure the underlying data structure. The effect is that the embeddings no longer represent the true cluster structure. We therefore propose a new robust spectral clustering algorithm that maps each high-dimensional feature vector onto a low-dimensional vector space. Robustness is achieved by posing the locality preserving feature mapping problem in form of a ridge regression task that is solved with a penalized M-estimation approach. An unsupervised penalty parameter selection strategy is proposed using the Fiedler vector, which is the eigenvector associated with the second smallest eigenvalue of a connected graph. More precisely, the penalty parameter is selected, such that, the corresponding Fiedler vector is ∆-separated with a minimum information loss on the embeddings. The method is benchmarked against popular embedding and spectral clustering approaches using real-world datasets that are corrupted by outliers.


For details, see:

[1] A. Taştan, M. Muma and A. M. Zoubir, “Robust Spectral Clustering: A Locality Preserving Feature Mapping Based on M-estimation," in Proc. 29th European Signal Process. Conf., pp. 851-855, 2021.

The codes can be freely used for non-commercial use only. Please make appropriate references to our article.

NOTE : This code requires an additional function 'whub' which is available in :

https://github.com/RobustSP/toolbox/blob/master/codes/07_AuxiliaryFunctions/whub.m

Further, it may require "enet.m" and "SoftThresh.m" functions if the preferred similarity measure is the elastic net similarity. The codes are, respectively, available in :

https://github.com/RobustSP/toolbox/blob/master/codes/02_Regression/enet.m

and

https://github.com/RobustSP/toolbox/blob/master/codes/07_AuxiliaryFunctions/SoftThresh.m

Lastly, it may require an additional function if the matrix XX in "Robust_Regularized_Locality_Preserving_Indexing.m" is rank deficient. An example function that is named "RankDeficientFastDecomposition.m" code is available in:

[2] P. Courrieu, “Fast computation of Moore-Penrose inverse matrices.” Online-Edition: https://arxiv.org/abs/0804.4809, 2008


