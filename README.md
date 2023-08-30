# generalizable-training-algorithms
Generalizable Training Algorithms for Deep Learning-Based Image Classification

## Abstract
In this project, we review and compare various existing neural network training algorithms. We find that all of the algorithms can successfully optimize the training loss function but they perform differently on unseen data points. We conduct experiments to confirm that Adam and other adaptive moment methods can minimize the training cost function faster than Stochastic Gradient Descent with Momentum (SGDM) for image classification tasks.  However, the test accuracy of Adam is significantly worse than SGDM's. We also reproduce Padam, a recently proposed algorithm that combines Adam with SGDM to achieve the best from both by introducing a partial adaptive parameter. 

Inspired by the existing algorithms, we propose a new class of algorithms by combining Adam and SGDM. This new class of algorithms includes Adam Switch SGDM and Linear Combination of SGDM and Adam (LCSA). LCSA further contains LCSA with Constant Weighting (LCSA-CW), LCSA with Discontinuous Dynamic Weighting (LCSA-DDW), and LCSA with Continuous Dynamic Weighting (LCSA-CDW).

We demonstrate via experiments that with proper hyperparameter tuning, LCSA-DDW can achieve a better test accuracy than SGDM and maintain a fast training convergence rate as Adam. Besides, LCSA-CW and LCSA-CDW could achieve better test accuracy than Adam while maintaining a fast training convergence rate. 



## [Read the full Paper](https://docs.google.com/document/d/1Qz9MVAgev6bTuz0uyQkxtq-2g2HMvoOhtkgmMjWvukQ/edit?usp=sharing)
