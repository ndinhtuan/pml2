## 2.3.3 Inverse problem *
Probability theory is concerned with predicting a distribution over outcomes $y$ given knowledge (or assumptions) about the state of the world, $h$. By contrast, **inverse probability** is concerned with inferring the state of the world from observation of outcomes. We can think of this as inverting the $h\rightarrow y$ mapping. 

For example, consider trying to infer a 3d shape $h$ from a 2d image $y$, which is a classic problem in **visual scene understanding**. Unfortunately, this is a fundamentally **ill-posed** problem.

To tackle such **inverse problems**, we can use Bayes' rule to compute the posterior, $p(h|y)$, which gives a distribution over possible states of the world. This requires specifying the **forwards model**, ```p(y|h)```, as well as a prior ```p(h)```, which can be used to rule out (or downweight) implausible world states. (The author will discuss detail in book3 edition publisded in 2023)

### Some related work in the Deep learning fields
This problem is very high related in different challenges in Deep learning such as monocular depth estimation, 3D object reconstruction. One example is "Deep MANTA: A Coarse-to-fine Many-Task Network for joint 2D and 3D vehicle analysis from monocular image" in CVPR2017, the authors try to reconstruct 3D car shape from a single image.

Prior knowledge ```p(h)``` the authors use is restricted 3D shape of car based on wireframe of car having some. From that we can reduce the value space of "world states".

About forwards model, ```p(y|h)``` many monocular depth estimation paper use some algorithm based on **projective geometry**, and in the ill-posed we use some inverse version of method to solve the challenges.

## 2.5.4 Log-sum-exp trick 
Suppose we want to compute the normalized probability $p_c=p(y=c|\mathbb{x})$, which is given by

$$p_c=\frac{e^{a_c}}{Z(\mathbb{a})} = \frac{e^{a_c}}{\sum^{C}_{c^{'}=1}e^{a_{c^{'}}}}$$

where $\mathbb{a} = f(\mathbb{x}, \mathbb{\theta})$ are the logits. We might encounter problems when computing the **partition function** Z. For example, suppose we have 3 classes, with logits $\mathbb{a}=(0,1,0)$. Then we find $Z=e^0 + e^1+e^0 = 4.71$. But now suppose $\mathbb{a}=(1000, 1001, 1000)$; we find $Z=\infin$, since on a computer even using 64 bit precision, ```np.exp(1000)=inf```. Similarly, suppose $\mathbb{a}=(-1000, -999, -1000)$; now  we find $Z=0$. To avoid numerical problems, we can use the following identity:
$$\log\sum^{C}_{c=1}\exp(a_{c})=m+\log\sum^{C}_{c=1}\exp(a_{c}-m)$$

This holds for any $m$. It is common to use $m=\max_c a_c$ which ensures that the largest value exponentiate will be zero, so you will definitely not overflow, and even if you underflow, the answer will be sensible. This is known as the **log-sum-exp trick**. We use this trick when implementing the **lse** function:
$$lse(\mathbb{a})\triangleq \log\sum^{C}_{c=1}\exp(a_{c})$$
We can use this to compute the probabilities from the logits:
$$p(y=c|\mathbb{x}) = \exp(a_c-lse(\mathbb{a}))$$

We can then pass this to the cross-entropy loss.

### Some practical application in Deep learning papers

In paper "Provable Guarantees for Understanding Out-of-Distribution Detection" published in AAAI-22, In **Remark 2** the authors proposed using ```log-sum-exp``` over Mahalanobis distances (compared with Lee et al. 2018b), instead of taking the ```maximum``` Mahalanobis distance. This was motivated in the paper where taking ```log-sum-exp```would be aligned with likelihood (w.r.t feature space), whereas ```max``` is not exact in theory