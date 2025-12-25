## 一、单变量函数收敛理论
### 1.1 二次型函数分析
考虑单变量二次型目标函数：
$$
E(w) = \frac{1}{2}aw^2 + bw + c
$$
其梯度下降更新公式为：
$$
w^{(k+1)} = w^{(k)} - \eta \frac{dE(w^{(k)})}{dw}
$$

**收敛性分析：**
在点 $w^{(k)}$ 处进行泰勒展开至二阶：
$$
E(w) = E(w^{(k)}) + E'(w^{(k)})(w - w^{(k)}) + \frac{1}{2}E''(w^{(k)})(w - w^{(k)})^2
$$

根据牛顿法，该二次近似的最小值点为：
$$
w_{min} = w^{(k)} - [E''(w^{(k)})]^{-1} E'(w^{(k)})
$$

由此可得**最优学习率**：
$$
\eta_{opt} = [E''(w^{(k)})]^{-1} = a^{-1}
$$

**收敛行为分类：**
- **$\eta < \eta_{opt}$**：算法单调收敛至最小值
- **$\eta_{opt} < \eta < 2\eta_{opt}$**：振荡收敛至最小值
- **$\eta > 2\eta_{opt}$**：算法发散

![收敛行为示意图](img/Pasted%20image%2020251222195035.png)

### 1.2 非二次函数分析
对一般目标函数进行二阶泰勒展开：
$$
E(w) \approx E(w^{(k)}) + \frac{dE(w^{(k)})}{dw}(w - w^{(k)}) + \frac{1}{2}\frac{d^2E(w^{(k)})}{dw^2}(w - w^{(k)})^2
$$

局部最优学习率近似为：
$$
\eta_{opt} \approx \left[\frac{d^2E(w^{(k)})}{dw^2}\right]^{-1}
$$

**收敛条件**：$\eta < 2\eta_{opt}$

---

## 二、多变量函数优化理论
### 2.1 二次型函数（对角矩阵情形）
考虑参数向量 $\mathbf{w} = [w_1, w_2, ..., w_N]^T$，目标函数为：
$$
E(\mathbf{w}) = \frac{1}{2}\mathbf{w}^T\mathbf{A}\mathbf{w} + \mathbf{b}^T\mathbf{w} + c
$$

当 $\mathbf{A}$ 为对角矩阵时：
$$
E(\mathbf{w}) = \sum_{i=1}^N \left(\frac{1}{2}a_{ii}w_i^2 + b_iw_i\right) + c
$$

各维度独立，最优学习率分别为：
$$
\eta_{opt,i} = \left[\frac{\partial^2 E}{\partial w_i^2}\right]^{-1} = \lambda_{ii}^{-1}
$$

**关键结论：**
1. 采用统一步长 $\eta$ 时，收敛要求：$\eta < 2\min_i\{\eta_{opt,i}\}$
2. 定义**条件数**：
   $$
   \kappa = \frac{\lambda_{\min}}{\lambda_{\max}} = \frac{\min_i\{\eta_{opt,i}\}}{\max_i\{\eta_{opt,i}\}}
   $$
   条件数越小，收敛速度越慢

### 2.2 坐标变换方法
为解决不同维度最优学习率差异问题，进行坐标变换：
$$
\hat{\mathbf{w}} = \mathbf{S}\mathbf{w}, \quad \mathbf{S} = \mathbf{A}^{1/2}
$$

变换后目标函数简化为：
$$
E(\hat{\mathbf{w}}) = \frac{1}{2}\hat{\mathbf{w}}^T\hat{\mathbf{w}} + \hat{\mathbf{b}}^T\hat{\mathbf{w}} + c
$$

此时梯度变换关系为：
$$
\nabla_{\hat{\mathbf{w}}}E = \nabla_{\mathbf{w}}E \cdot \mathbf{A}^{-1/2}
$$

参数更新公式变为：
$$
\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \eta\mathbf{A}^{-1}\nabla_{\mathbf{w}}E(\mathbf{w}^{(k)})^T
$$

**最优学习率**：$\eta = 1$

![坐标变换效果示意图](img/Pasted%20image%2020251222203039.png)

### 2.3 牛顿二阶方法（一般函数）
#### 2.3.1 基本原理
对一般函数进行二阶泰勒展开：
$$
E(\mathbf{w}) \approx E(\mathbf{w}^{(k)}) + \nabla E(\mathbf{w}^{(k)})^T(\mathbf{w} - \mathbf{w}^{(k)}) + \frac{1}{2}(\mathbf{w} - \mathbf{w}^{(k)})^T\mathbf{H}_E(\mathbf{w}^{(k)})(\mathbf{w} - \mathbf{w}^{(k)})
$$

参数更新公式：
$$
\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \eta\mathbf{H}_E^{-1}(\mathbf{w}^{(k)})\nabla E(\mathbf{w}^{(k)})^T
$$

其中 $\eta = 1$ 为最优值（满足 $\eta < 2$）。

#### 2.3.2 几何解释
牛顿方法等价于：
1. 在当前点用二次型局部拟合目标函数
2. 对该二次型采用最优步长直接优化至最小值
3. 本质上是对Hessian矩阵进行特征分解后的坐标变换

**数学表示**：
$$
\mathbf{H} = \mathbf{U}^T\boldsymbol{\Lambda}\mathbf{U}, \quad \hat{\mathbf{w}} = \mathbf{U}\mathbf{w}
$$

![牛顿方法收敛示意图](img/Pasted%20image%2020251222204302.png)

#### 2.3.3 局限性
1. **计算复杂度高**：Hessian矩阵求逆为 $O(N^3)$ 复杂度
2. **稳定性问题**：在非凸区域，Hessian矩阵可能非正定，导致算法发散
3. **局部极值问题**：固定学习率容易陷入局部最小值

---

## 三、自适应步长方法
### 3.1 Rprop方法
**核心思想**：根据梯度符号变化调整步长

**更新规则**：
- 当前梯度与上一步梯度同号：增大步长
  $$
  \Delta w^{(k)} = \alpha \Delta w^{(k-1)}, \quad \alpha > 1
  $$
- 当前梯度与上一步梯度异号：回退并缩小步长
  $$
  w^{(k)} = w^{(k-1)}, \quad \Delta w^{(k-1)} = \beta \Delta w^{(k-2)}, \quad \beta < 1
  $$

### 3.2 QuickProp方法
**基本假设**：各参数维度相互独立

**更新公式**：
$$
w_i^{(k+1)} = w_i^{(k)} - \left[\frac{\partial^2 E}{\partial w_i^2}\right]^{-1} \frac{\partial E}{\partial w_i}
$$

**二阶导数近似**（有限差分法）：
$$
\frac{\partial^2 E}{\partial w_i^2} \approx \frac{\Delta w_i^{(k-1)}}{E'(w_i^{(k)}) - E'(w_i^{(k-1)})}
$$

### 3.3 动量法
**基本思想**：使用历史更新方向的指数移动平均

**更新公式**：
$$
\Delta \mathbf{W}^{(k)} = \beta\Delta \mathbf{W}^{(k-1)} - \eta\nabla L(\mathbf{W}^{(k-1)})^T
$$

**典型参数**：$\beta = 0.9$

**优势**：
- 平滑收敛方向，抑制振荡
- 在一致方向上积累动量，加速收敛

![动量法收敛示意图](img/Momentum.png)

### 3.4 Nesterov加速梯度法
**改进思路**：先沿动量方向"前瞻"，再计算该位置的梯度

**更新公式**：
$$
\Delta \mathbf{W}^{(k)} = \beta\Delta \mathbf{W}^{(k-1)} - \eta\nabla L(\mathbf{W}^{(k-1)} + \beta\Delta \mathbf{W}^{(k-1)})^T
$$

**优势**：比标准动量法具有更好的理论收敛性质

![Nesterov方法收敛示意图](img/nesterov.png)

---

## 四、SGD收敛性理论
### 4.1 收敛定义与条件
**收敛定义**：
$$
|f(\mathbf{w}^{(k)}) - f(\mathbf{w}^*)| < \epsilon
$$

**充分条件**（理论保证）：
$$
\sum_{k=1}^{\infty} \eta_k = \infty, \quad \sum_{k=1}^{\infty} \eta_k^2 < \infty
$$

**实践策略**：
- 强凸函数：采用 $1/k$ 衰减策略可达最优收敛速度
- 一般情况：需要适当衰减学习率保证收敛

### 4.2 收敛速度比较

| 算法类型 | 函数类别 | 误差衰减 | 收敛速度 | $\epsilon$-精度所需迭代次数 |
|---------|---------|---------|---------|---------------------------|
| **SGD** | 强凸函数 | $1/k$ | $O(1/k)$ | $O(1/\epsilon)$ |
| | 一般凸函数 | $1/\sqrt{k}$ | $O(1/\sqrt{k})$ | $O(1/\epsilon^2)$ |
| **Batch GD** | 强凸函数 | 指数衰减 | $O(\log(1/\epsilon))$ | $O(\log(1/\epsilon))$ |
| | 一般凸函数 | - | $O(1/\epsilon)$ | $O(1/\epsilon)$ |
| **Minibatch GD** | 一般凸函数 | $O(1/\sqrt{bk} + 1/k)$ | 优于SGD | 依赖batch大小 $b$ |

### 4.3 算法选择指导

**SGD优势**：
- 每次迭代计算量小
- 适合大规模数据集
- 总体收敛速度可能更快

**Batch GD特点**：
- 每次迭代需要完整数据集
- 迭代次数少但每次迭代代价高

**Minibatch GD实践建议**：
1. **平衡效率与稳定性**：batch大小适中（通常32-256）
2. **并行计算友好**：适合GPU等硬件加速
3. **噪声与收敛平衡**：适当噪声有助于跳出局部极小值

**实际应用考虑**：
- 目标函数通常非凸，mini-batch表现更稳健
- 需要根据具体问题和计算资源调整batch大小
- 学习率衰减策略对最终收敛至关重要
