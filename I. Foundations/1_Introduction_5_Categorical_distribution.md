## 2.5.4 Log-sum-exp trick 
Suppose we want to compute the normalized probability $p_c=p(y=c|\mathbb{x})$, which is given by
$$p_c=\frac{e^{a_c}}{Z(\mathbb{a})} = \frac{e^{a_c}}{\sum^{C}_{c^{'}=1}e^{a_{c^{'}}}}$$