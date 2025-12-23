## 单变量函数收敛理论
最小化 $E=\frac{1}{2}aw^2+bw+c$，梯度下降$w^{(k+1)}=w^{(k)}-\eta \frac{dE(w^{(k)})}{dw}$ ，能否收敛到目标函数最小值点依赖于步长（学习率）$\eta$的取值。
泰勒级数展开有：
$$E(w)=E(w^{(k)}+E'(w^{k})(w-w^{(k)})+\frac{1}{2}E''(w^{(k)})(w-w^{(k)})^2$$
根据牛顿方法，目标函数E最小值除w取值为$w_{min}=w^{(k)}-E''(w^{(k)})^{-1}E'(w^{(k)})$，则最优步长为$$\eta_{opt}=E''(w^{(k)})^{-1}=a^{-1}$$
如果$\eta < \eta_{opt}$，算法会单调收敛最小值；$\eta_{opt}<\eta<2\eta_{opt}$时，振荡收敛至最小值；$\eta>2\eta_{opt}$时，算法发散。
![[Pasted image 20251222195035.png]]
![alt text](<figure/Pasted image 20251222195035.png>)
### 非二次函数
目标函数泰勒级数展开至二次项：
$$E\approx E(w^{(k)})+\frac{dE(w^{(k)})}{dw}(w-w^{(k)})+\frac{1}{2}\frac{d^2E(w^{(k)})}{dw^2}(w-w^{(k)})^2+...$$
同样可以得到最优学习率为$$\eta_{opt}=\left(\frac{d^2E(w^{(k)})}{dw^2}\right)^{-1}$$
收敛要求$\eta<2\eta_{opt}$ 。

## 多变量函数优化
### 二次型函数
存在多个参数需要优化$\textbf{w}=[w_1,w_2,...,w_N]$，目标函数为二次型时
$$E=\frac{1}{2}\textbf{w}^T\textbf{Aw}+\textbf{w}^T\textbf{b}+c$$
假设$\textbf{A}$为对角矩阵，有$E=\sum_{i}\left (\frac{1}{2}a_{ii}w_{i}^2+b_iw_i\right)+c$，这时输入变量不相关，可以分别计算各自的最优步长$$\eta_{opt,i}=\left( \frac{\partial^2E(w_i^{(k)})}{\partial w_i^2}\right)^{-1}=\lambda_{ii}^{-1}$$
其中最优步长等于矩阵$\mathbf{A}$特征值的倒数，收敛要求$\eta<2min_i\{\eta_{opt,i}\}$（**假设采用统一步长**）。
根据特征值大小定义条件数$$\frac{\lambda_{min}}{\lambda_{max}}=\frac{min_i\{\eta_{opt,i}\}}{max_i\{\eta_{opt,i}\}}$$条件数越小，收敛速度越慢。
### 坐标变换
对于不同参数最优学习率差异较大的情况，可以通过变化坐标系归一化，即设$\hat{w}=\textbf{Sw}$，使得目标函数变为$E=\frac{1}{2}\hat{\textbf{w}}^T\hat{\textbf{w}}+\hat{\textbf{b}}^T\hat{\textbf{w}}+c$ ，其中$S=\textbf{A}^{0.5},\hat{\textbf{b}}=\textbf{A}^{-0.5}\textbf{b}$，此时$\nabla_{\hat{\textbf{w}}}E=\nabla_{\textbf{w}}E\cdot \textbf{A}^{-0.5}$, 则参数更新变为：
$$\textbf{w}^{(k+1)}=\textbf{w}^{(k)}-\eta\textbf{A}^{-1}\nabla_{\textbf{w}}E(w^{(k)})^T$$
其中最优学习率$\eta=1$。
![[Pasted image 20251222203039.png]]
![alt text](<figure/Pasted image 20251222203039.png>)
### 一般函数（牛顿二阶方法）
对于一般的目标函数，采用泰勒级数展开：
$$E(\textbf{w})\approx E(\textbf{w}^{(k)})+\nabla_{\textbf{w}}E(\textbf{w}^{(k)})(\textbf{w}-\textbf{w}^{(k)})+\frac{1}{2}(\textbf{w}-\textbf{w}^{(k)})^TH_E(\textbf{w}^{(k)})(\textbf{w}-\textbf{w}^{(k)})$$
对左边进行正交化之后得到参数更新公式
$$\textbf{w}^{(k+1)}=\textbf{w}^{(k)}-\eta\textbf{H}_E(\textbf{w}^{(k)})^{-1}\nabla_{\textbf{w}}E(w^{(k)})^T,\eta=1<2$$

该坐标变换相当于对Hessian矩阵进行特征分解$\mathbf{H}=\mathbf{U^T\Lambda  U}$，使用特征向量组成的矩阵进行坐标系旋转$\mathbf{\hat{w}}=\mathbf{Uw}$ ，使用特征值$\lambda_i$进行缩放坐标轴。

牛顿方法等价于在$\textbf{w}^{(k)}$附近使用二次型拟合，随后对该二次型采用最优步长优化至最小值。
![[Pasted image 20251222204302.png]]
![alt text](<figure/Pasted image 20251222204302.png>)
**缺点：
	海森矩阵纬度高，求逆计算复杂度高
	在非凸区域 Hessian 可能非正定，导致算法发散
	学习率$\eta<2$固定，容易陷入局部最小值**
### 非固定步长
#### Rprop 方法
*核心思想：如果梯度在该步于上一步同号，则增大步长$\Delta w^{(k)}=\alpha \Delta w^{(k-1)}, \alpha>1$ ；否则回退到上一步并缩小步长$\Delta w^{(k-1)}=\beta \Delta w^{(k-2)},\beta<1$。*
#### QuickPorp 方法
基于牛顿二阶方法，假设每个参数维度独立：
$$w_i^{k+1}=w_i^k-E''(w_i^k|w_j^k,j\neq i)^{-1}E'(w_i^k|w_j^k,j\neq i)$$
同时采用有限差分近似计算二阶导数：
$$w_i^{k+1}=w_i^k-\frac{\Delta w_i^{k-1}}{E'(w_i^{(k)})-E'(w_i^{k-1})}E'(w_i^k)$$
#### 动量法
计算历史步长的移动平均，收敛平滑的方向保持较大步长，振荡的方向正负更新抵消，公式为
$$\Delta W^k = \beta\Delta W^{k-1}-\eta\nabla_{W}Loss(W^{k-1})^T$$
其中典型值为$\beta=0.9$。
![[Pasted image 20251222211136.png]]
![alt text](<figure/Pasted image 20251222211136.png>)
#### Nestorov’s 加速梯度
首先延续上一部更新，之后计算该位置的梯度，最后合并两部更新，公式如下：
$$\Delta W^k = \beta\Delta W^{k-1}-\eta\nabla_{W}Loss(W^{k-1}+\beta\Delta W^{k-1})^T$$
![[Pasted image 20251222211444.png]]
![alt text](<figure/Pasted image 20251222211444.png>)

## SGD 的收敛性
收敛定义为$|f(w^k)-f(w^*)|<\epsilon$，收敛必须保证学习率衰减；
**充分条件:** 理论上保证全局收敛的充分条件是学习率序列满足 $\sum \eta_k = \infty$ 且 $\sum \eta_k^2 < \infty$ ；
对于强凸函数，SGD 使用 $1/k$ 的学习率衰减策略可以达到最优收敛速度；

### 收敛速度
SGD：
强凸函数的误差与迭代次数k成反比
$$|f(w^k)-f(w^*)|=\frac{1}{k}|f(w^0)-f(w^*)|$$

因此收敛速度为$O(1/k)$，迭代$O(1/\epsilon)$次可以达到误差$\epsilon$；
一般凸函数的误差与$\sqrt{k}$成反比，收敛速度为$O(1/\sqrt{k})$，需要迭代$O(1/\epsilon^2)$次收敛；

Batch GD：
对于强凸函数，误差指数下降：
$$|f(w^k)-f(w^*)|=c^k|f(w^0)-f(w^*)|$$
收敛需要$O(log(1/\epsilon))$次迭代；
一般的凸函数需要$O(1/\epsilon)$次迭代。
尽管Batch GD需要的迭代次数少，但是每次迭代需要计算整个数据集，而SGD每次迭代很快，总体来看SGD收敛更快。

Minibatch GD：
对于一般的凸函数，minibatch 大小为b，收敛速率为$O(1/\sqrt{bk}+1/k)$，比SGD小；
实际应用中，目标函数一般非凸，mini-batch更加有效，同时方便批量并行计算。
