## Introduction
Chapter 1
Gradient Desent && Backpropagation

## GD in Mathematical Optimization
### 问题定义
*   **目标**：寻找目标函数 $ J(\theta) $ 的极小值点（通常为局部极小值）及其对应的参数 $\theta^*$，即 $\theta^* = \arg\min_{\theta} J(\theta)$

![alt text](img/math.png)

*   **挑战**：目标函数通常具有高维、非凸、非线性的复杂特性，且可能伴有噪声。导致无法通过直接解析（如令导数为零）的方法求得全局最优解，必须依赖迭代数值优化算法。

---

### 梯度下降（SGD, Stochastic Gradient Descent）
基于一阶导数的经典迭代优化算法。
**核心思想**：沿着目标函数在当前点的梯度反方向（即函数值下降最快的方向）逐步调整参数，从而逼近函数的局部极小值。

**算法过程**：
1.  **初始化**：随机选择或指定一组参数的初始值 $\theta_0$，设定学习率 $\eta$（步长）和迭代终止条件（如最大迭代次数或梯度范数阈值）。
2.  **迭代更新**：对于第 $k$ 次迭代（$k = 0, 1, 2, ...$）：
    *   **计算梯度**：计算目标函数在当前参数 $\theta_k$ 处的梯度 $\nabla J(\theta_k)$。
    *   **更新参数**：沿梯度反方向更新参数：$\theta_{k+1} = \theta_k - \eta \cdot \nabla J(\theta_k)$。
    *   **检查收敛**：判断是否满足终止条件（如梯度足够小、$J(\theta)$ 变化不明显或达到最大迭代次数）。若满足，则停止迭代，输出当前 $\theta_{k+1}$ 作为近似解；否则，继续进行下一轮迭代。

**相关挑战**：
*   **学习率选择**：固定学习率难以适应训练全过程。
*   **地形问题**：不同参数方向梯度尺度差异巨大（病态条件数，如沟壑/平原），导致收敛缓慢。
*   **局部最优与鞍点问题**：在高维空间，鞍点比局部极值点更常见。

---

### 算法演进
1.  **引入动量（Momentum）**
    *   **核心思想**：模拟物理中的动量，当前更新方向不仅取决于当前梯度，还累积历史梯度的指数加权平均。
    *   **作用**：
        *   **加速收敛**：在梯度方向一致的维度上加速。
        *   **抑制震荡**：通过动量平滑掉不一致的梯度噪声，帮助穿越狭窄的“沟壑”。
    *   **SGD with Momentum**
        引入速度变量 $ \mathbf{v}_t $，其更新规则如下：
        1. **计算动量（速度）**：
        $$
        \mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \mathbf{g}_t
        $$
        其中：
        * $ \mathbf{g}_t $ 是当前时刻的梯度。
        * $ \beta $ 是动量系数（通常设为0.9），控制历史速度的衰减程度。
        * $ \mathbf{v}_t $ 本质上是历史梯度的一个指数加权移动平均，它积累了之前更新的方向。
        2. **参数更新**：
        $$
        \theta_t = \theta_{t-1} - \eta \mathbf{v}_t
        $$
        其中 $ \eta $ 是学习率。
    *   **SGD with Nesterov Acceleration**
        *   **核心思想**：**前瞻一步**
            * 不像标准动量，先计算当前点的梯度然后叠加动量
            * 而是先让参数沿着当前累积的动量方向（$ \beta \mathbf{v}_{t-1} $）进行**临时更新** ，得到前瞻位置（$\theta_{t-1} - \eta \beta \mathbf{v}_{t-1}$）。然后，**在前瞻位置计算梯度**，用此梯度校正当前的动量更新。
        *   **数学表达**：
            1. **计算前瞻位置的梯度**：
            $$
            \mathbf{g}_t^{lookahead} = \nabla J(\theta_{t-1} - \eta \beta \mathbf{v}_{t-1})
            $$
            2. **更新速度（动量）**：
            $$
            \mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \mathbf{g}_t^{lookahead}
            $$
            3. **参数更新**：
            $$
            \theta_t = \theta_{t-1} - \eta \mathbf{v}_t
            $$
        *   **e.g.**：当参数在动量作用下冲向谷底时，标准动量会“冲过头”才根据谷底的梯度调整；而NAG会提前“看到”谷底的坡度，从而提前减速并更准确地转向。
        *   使得NAG在理论上**具有更优的收敛率**，在实践中对于循环神经网络（RNN）等模型的训练往往表现更好，能更有效地处理损失函数中的“沟壑”地形。

2.  **自适应学习率（Per-parameter Learning Rate）**
    *   **核心思想**：为网络中**每一个参数**单独调整学习率。根据该参数的历史梯度信息，自动放大或缩小其更新步长。
    *   **RMSProp** (Root Mean Square Propagation)
    引入衰减系数，**只关注近期梯度历史**，淡化遥远过去梯度的影响。从而解决了学习率单调下降的问题，使训练能够持续进行。
    1.  **计算梯度平方的指数移动平均**：
        $$
        E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1 - \rho) \cdot g_t \odot g_t
        $$
        不累积全部历史梯度平方和，而是计算一个**指数衰减的移动平均**。
        *   $ \rho $ 是衰减率（通常设为0.9）
        *   $ E[g^2]_t $ 可以理解为**近期梯度平方的期望估计**，赋予算法“短期记忆”。
    2.  **参数更新**：
        $$
        \theta_{t} = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} \odot g_t
        $$
    *   **自适应机制**：对于某个参数，如果其历史梯度 $ \sqrt{E[g^2]_t} $ 很大（更新频繁或幅度大），那么其对应的缩放因子就会变小，从而**减小该参数的实际学习步长**。反之，对于历史梯度小的稀疏参数，其有效学习率相对较大。

---

### 主流
1. **Adam** (Adaptive Moment Estimation)
    **同时结合了动量（Momentum）和RMSProp的思想**，通过自适应调整每个参数的学习率来加速收敛并提升训练稳定性。
    1.  **计算梯度的一阶矩估计（动量项）**：
        $$
        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
        $$
        其中
        - $ g_t $ 是当前时间步 $ t $ 的梯度
        - $ m_{t-1} $ 是上一时间步的动量向量
        - $ \beta_1 $是衰减率（通常=0.9）

        相当于对梯度做**指数移动平均**，保留历史梯度方向的信息
    2.  **计算梯度的二阶矩估计（自适应项）**：
        $$
        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
        $$
        其中
        - $ g_t^2 $ 表示梯度的逐元素平方
        - $ \beta_2 $是另一个衰减率（通常=0.999）
        
        相当于对梯度平方做**指数移动平均**，反映**各参数历史梯度量级的变化幅度**。量级（对应 $ v_t $）大的参数通常意味着其更新不稳定或梯度本身较大，后续作自适应调节。
    3.  **偏差修正**：
        $$
        \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \ \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
        $$
        由于 $ m_t $ 和 $ v_t $ 从零向量开始进行指数移动平均，在训练早期（$ t $ 较小时）会严重偏向于零，导致更新步长被低估，使得初期收敛异常缓慢。
        - **期望分析**：
            - 假设梯度 $ g_t $ 平稳，其期望为 $ \mathbb{E}[g_t] = \mu $，方差为 $ \sigma^2 $。
            - 动量项 $ m_t $ 的期望（**未修正**）：
            $$
            \mathbb{E}[m_t] = \mathbb{E}[(1 - \beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i] = (1 - \beta_1^t) \mu
            $$
            - 同理，$ \mathbb{E}[v_t] = (1 - \beta_2^t) \mathbb{E}[g_t^2] $。
        - **修正方法**：
            - 将 $ m_t, v_t $ 除以 $ 1 - \beta^t $：
            $$
            \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
            $$
            - 修正后，$ \mathbb{E}[\hat{m}_t] = \mu $，$ \mathbb{E}[\hat{v}_t] = \mathbb{E}[g_t^2] $，消除了初期偏差。
    4.  **参数更新**：
        $$
        \theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
        $$
        其中
        - $ \eta $ 是初始学习率
        - $ \epsilon $ 极小（通常≈$ 10^{-8} $），防止除以零
        - $ \hat{m}_t $ 提供**动量方向**
        - $ \sqrt{\hat{v}_t} $ 起到**逐参数自适应缩放学习率**的作用：对于历史梯度平方和较大的参数，其步长会被缩小；反之则步长相对较大

2.  **AdamW** (Adam with Weight Decay)
    **正则化(Regularization)**:
    *   **核心思想**：在模型的损失函数中添加一个**惩罚项**，约束模型参数的大小，鼓励模型学习更简单、更通用的模式，而不是依赖少数几个特征或极端权重。
    *   $L_{1}$：添加模型**权重绝对值之和**乘以正则化强度系数 λ
        $$
        L_{1} = L(w) + \lambda \sum_{i=1}^{n} |w_i|
        $$
    *   $L_{2}$：添加模型**权重平方和**乘以正则化强度系数 λ
        $$
        L_{2} = L(w) + \lambda \sum_{i=1}^{n} w_i^2
        $$
    
    **将权重衰减（Weight Decay，即L2正则化项）从梯度计算中解耦出来**，直接、独立地应用于参数更新，而非作为损失函数的一部分影响梯度。在标准Adam优化器基础上的一个重要改进。
    1.  **计算梯度的一阶矩估计（动量项）**：与Adam完全相同
    2.  **计算梯度的二阶矩估计（自适应项）**：与Adam完全相同
    3.  **偏差修正**：相同
    4.  **参数更新**：
        $$
        \theta_{t} = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \color{blue}{+ \lambda \theta_{t-1}} \right)
        $$
        由两部分组成：
        -   **自适应梯度更新**：$ \eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $，这部分与Adam一致
        *   **解耦的权重衰减**：$ \color{blue}{\eta_t \lambda \theta_{t-1}} $，这是一个**直接、与梯度无关的收缩项**。它独立于自适应学习率机制，以一个固定的速率 $ \lambda $（权重衰减系数）将参数向零收缩。**以实现正则化**

    **与标准Adam（耦合权重衰减）的对比**：
    - 在标准Adam中，权重衰减通常被加入损失函数，即梯度 $ g_t $ 中已经包含了 $ \lambda \theta_{t-1} $ 项。
    - 意味着**权重衰减的效果会被自适应学习率（$ \sqrt{\hat{v}_t} $）所缩放**，导致对于梯度较大的参数，其正则化强度反而被削弱。
    - AdamW通过解耦，确保了**权重衰减的效果是稳定且一致的**，不受参数梯度历史量级的影响。

    **优势总结**：
    - AdamW保留了Adam的所有优点——**动量带来的方向稳定性和自适应学习率带来的各参数步长灵活性**。
    - 通过**解耦权重衰减**，提供了更纯粹、更可控的正则化。使模型在追求高性能的同时，能更好地避免过拟合。
    - 在大容量模型（如Transformer）的训练中，AdamW已成为获得更优泛化能力的首选优化器。

---

### Choose Optimizer
```mermaid
flowchart TD
    A[开始选择] --> B[数据形式]
    
    B -- 稀疏/高维数据<br>（如文本） --> C[Adam/AdamW]
    C --> D[模型参数量大小]
    
    B -- 稠密数据<br>（如图像） --> E[首要训练目标]
    
    E -- 追求极致泛化性能 --> F[SGD+Momentum]
    E -- 追求速度与稳定性  --> G[AdamW]
    
    F --> D
    G --> D
    
    D -- 数十亿参数 --> H[LAMB/Adafactor]
    D -- 参数量较小 --> I[保持当前选择]
```

#### 按数据类型
这是最关键的划分依据，直接决定是否使用自适应学习率优化器。

1. 稀疏/高维数据（文本/NLP、推荐系统、多分类任务）
- **适用优化器**：Adam/AdamW
- **核心原因**：数据特征频率差异大（如文本中高频词和稀有词），自适应优化器能为每个参数动态调整学习率：高频特征（梯度大）降低学习率，稀有特征（梯度小）提高学习率，避免“一刀切”导致部分特征学不到。
- **示例场景**：训练大词汇量语言模型（如BERT、GPT）、推荐系统用户-物品交互矩阵学习、文本分类任务。

2. 稠密数据（计算机视觉、图像分类、稠密特征回归）
- **适用优化器**：SGD+Momentum（首选）、AdamW（次选）
- **核心原因**：
  - 稠密数据（如图像像素、连续特征）梯度分布相对均匀，SGD的“噪声更新”能帮助模型跳出尖锐极小值，找到泛化性更好的平坦极小值。
  - 虽然SGD收敛慢，但配合学习率调度（如余弦退火、步衰减）和BatchNorm，最终精度常高于Adam（尤其ImageNet等大型视觉任务）。
- **示例场景**：ResNet等CNN模型训练、ImageNet图像分类、基于稠密特征的回归任务。

#### 按训练优先级
1. 优先训练速度和易用性（快速验证、大规模训练）
- **适用优化器**：Adam、AdamW（首选）
- **核心原因**：
  - 收敛速度快，无需复杂调参，默认参数适配多数任务，能快速得到可用结果。
  - 对噪声梯度、稀疏梯度鲁棒性强，适合LLM、Transformer、RNN/LSTM等复杂模型（避免训练卡顿）。
  - 大规模训练（如LLM预训练）中，Adam/AdamW能稳定利用GPU/TPU并行计算，降低训练成本（训练成本达数百万美元时，快速收敛至关重要）。
- **示例场景**：LLM预训练（GPT、BERT）、快速验证模型架构、RNN/LSTM训练（处理长序列依赖）。

2. 优先极致泛化性能（学术竞赛、工业部署）
- **适用优化器**：SGD+Momentum（首选）、NAG（进阶）
- **核心原因**：
  - SGD的随机噪声相当于“隐式正则化”，能避免模型过拟合到训练数据的细节，测试集性能更稳定。
  - 配合学习率调度（如warmup+余弦衰减）、权重衰减、BatchNorm，可最大化泛化能力（许多SOTA视觉模型仍依赖此组合）。
- **注意**：需要更多调参成本（学习率、动量系数γ=0.9默认），且训练周期更长。

#### 按内存和硬件限制
1. 超大规模模型（数十亿参数，如GPT-3）
- **适用优化器**：Adafactor、LAMB（替代Adam/AdamW）
- **核心原因**：
  - Adam/AdamW需要存储两个与参数等大的矩向量（m和v），数十亿参数场景下内存占用爆炸；Adafactor通过因式分解二阶矩，大幅降低内存消耗（代价是收敛质量略有下降）。
  - LAMB支持超大数据量（32K/64K样本），通过层归一化更新，避免大批量训练时的梯度爆炸，适合硬件集群并行训练（如Google BERT大规模预训练）。
- **硬件适配**：TPU训练常用Adafactor，GPU集群大批量训练常用LAMB。

2. 内存极紧张（嵌入式设备、小显存GPU）
- **适用优化器**：SGD（首选）、SGD+Momentum（次选）
- **核心原因**：
  - 内存占用最小，仅需存储当前梯度，无需额外存储矩向量（Adam/Adagrad等需额外内存）。
  - 计算开销低，适合嵌入式设备、小显存GPU等资源受限场景。

---



## Backpropagation && GD in Neural Network
### Neural Network
1. **Feedforward Neural Network Intro**
    * 由大量相互连接的神经元组成的计算模型，通过层层非线性变换，能够学习输入与输出之间复杂的映射关系。
    * 信息单向流动，数据从输入层经过隐藏层最终到达输出层，没有循环或反馈连接
2.  **问题定义**
    *   **目标**：寻找一组最优权重参数，最小化损失函数，使模型预测最准确。
    *   **挑战**：高维、非凸、非线性的复杂优化问题，无法直接解析求解。
3.  **核心方法**
    *   **反向传播（Backpropagation）**：计算梯度。高效、精确地计算损失函数对百万乃至亿级权重的梯度（导数）。
    *   **梯度下降（Gradient Descent）**：更新模型参数。利用梯度信息，迭代地更新权重，引导模型向最优解移动。

---

### 模型损失函数 && 反向传播
**损失函数**：衡量模型预测与真实目标之间差异的标量函数，训练的目标就是最小化损失函数。  
如用于回归的均方误差（MSE）；用于分类的交叉熵损失（Cross-Entropy）

**反向传播核心步骤**：
1.  **前向传播**：输入数据通过网络层层传递，直至输出层得到预测值。此过程计算并保存每一层的中间激活值，同时计算最终的损失函数值。
2.  **反向传播**：从输出层开始，计算损失函数对输出层输入的梯度，然后利用链式法则，逐层向后计算损失函数对每一层权重和输入的梯度。这个过程中，中间结果的梯度可以被复用，大大提高了计算效率。最终，我们得到损失函数关于网络中所有权重参数的梯度向量 $\nabla_{\theta} L(\theta)$，其中 $\theta$ 代表全部参数。

---

### 反向传播Demo
#### 前向传播
##### 1. 线性层
$$
z = w \cdot x + b
$$
其中：
- $ x $：输入（标量）
- $ w $：权重（标量）
- $ b $：偏置（标量）
- $ z $：线性层输出（标量）

##### 2. 激活函数
$$
a = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

##### 3. 损失函数
$$
\text{loss} = -\left[ y \cdot \log(a) + (1 - y) \cdot \log(1 - a) \right]
$$
其中 $ y $ 是真实标签（0或1）。

#### 反向传播推导
##### 1. 计算损失对激活输出 $ a $ 的梯度
$$
\frac{\partial \text{loss}}{\partial a} = -\left[ \frac{y}{a} - \frac{1-y}{1-a} \right]
$$
化简得：
$$
\frac{\partial \text{loss}}{\partial a} = \frac{a - y}{a(1-a)}
$$

##### 2. 计算激活函数对 $ z $ 的梯度
$$
\frac{\partial a}{\partial z} = a(1-a)
$$

##### 3. 计算损失对线性输出 $ z $ 的梯度（链式法则）
$$
\frac{\partial \text{loss}}{\partial z} = \frac{\partial \text{loss}}{\partial a} \cdot \frac{\partial a}{\partial z}
$$
代入得：
$$
\frac{\partial \text{loss}}{\partial z} = \frac{a - y}{a(1-a)} \cdot a(1-a) = a - y
$$

##### 4. 计算损失对权重 $ w $ 的梯度
由于 $ z = x \cdot w + b $，有：
$$
\frac{\partial z}{\partial w} = x
$$
应用链式法则：
$$
\frac{\partial \text{loss}}{\partial w} = \frac{\partial \text{loss}}{\partial z} \cdot \frac{\partial z}{\partial w} = (a - y) \cdot x
$$

##### 5. 计算损失对偏置 $ b $ 的梯度
由于 $ z = x \cdot w + b $，有：
$$
\frac{\partial z}{\partial b} = 1
$$
应用链式法则：
$$
\frac{\partial \text{loss}}{\partial b} = \frac{\partial \text{loss}}{\partial z} \cdot \frac{\partial z}{\partial b} = a - y
$$

##### 6. 计算损失对输入 $ x $ 的梯度
$$
\frac{\partial z}{\partial x} = w
$$
$$
\frac{\partial \text{loss}}{\partial x} = \frac{\partial \text{loss}}{\partial z} \cdot \frac{\partial z}{\partial x} = (a - y) \cdot w
$$

#### 反向传播公式总结
| 参数 | 梯度公式 |
|------|----------|
| 损失对输出 $ z $ 的梯度 | $\displaystyle \frac{\partial \text{loss}}{\partial z} = a - y = \sigma(z) - y$ |
| 损失对权重 $ w $ 的梯度 | $\displaystyle \frac{\partial \text{loss}}{\partial w} = (a - y) \cdot x$ |
| 损失对偏置 $ b $ 的梯度 | $\displaystyle \frac{\partial \text{loss}}{\partial b} = a - y$ |
| 损失对输入 $ x $ 的梯度 | $\displaystyle \frac{\partial \text{loss}}{\partial x} = (a - y) \cdot w$ |

#### 参数更新公式（梯度下降）
使用学习率 $ \eta $ 更新参数：
$$
w \leftarrow w - \eta \cdot \frac{\partial \text{loss}}{\partial w}
$$
$$
b \leftarrow b - \eta \cdot \frac{\partial \text{loss}}{\partial b}
$$

---

### 梯度下降算法 with Batch Size
梯度下降的核心迭代公式为：
$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_{\theta} L(\theta_t)
$$
其中 $\eta$ 是学习率，控制更新的步长。  
根据计算梯度时所用数据量的不同，梯度下降主要有三种变体，它们在效率、稳定性和泛化能力上各有权衡：

1.  **（纯）随机梯度下降（SGD）**
    *   **原理**：每次迭代随机使用**一个样本**计算梯度并立即更新。
    *   **收敛速度**：对于一般凸函数，理论收敛速度为$O(1/\sqrt{k})$，k为迭代步数，需要迭代$O(1/\epsilon^2)$次收敛。
    *   **优点**：
        *   **速度极快**，更新频率高。
        *   **引入的噪声有助于跳出局部最优和尖锐极小点**，可能找到泛化更好的平坦解。
    *   **缺点**：**更新方向震荡剧烈**，收敛过程不稳定，难以精细调优。

2.  **批量梯度下降（Batch GD）**
    *   **原理**：每次迭代使用**全部训练数据**计算精确梯度。
    *   **收敛速度**，对于一般凸函数，理论上需要迭代$O(1/\epsilon)$次使得目标函数误差小于$\epsilon$。
    *   **优点**：梯度方差相比单样本缩小$1/N$，梯度方向稳定，理论收敛性好。
    *   **致命缺点**：
        *   **计算成本极高**：每步更新都需遍历全量数据，内存和计算无法承受。
        *   **易陷入局部最优**：更新方向过于“确定”，缺乏探索能力。
        *   **无法在线学习**。

3.  **小批量随机梯度下降（Mini-batch SGD）**
    *   **原理**：将数据划分为若干**小批量（Batch）**，每次使用一个批次的样本估计梯度。
    *   **收敛速度**：对于一般的凸函数，batch取为b时，收敛速率为$O(1/\sqrt{bk}+1/k)$。
    *   **为什么是“估计”？** 小批量梯度是全量数据真实梯度的**无偏但带噪声的估计**。
    *   **“小批量”的平衡艺术**：
        *   **效率**：相比全批量，计算开销小，能充分利用GPU并行计算。
        *   **稳定性**：相比单样本，梯度的方差缩小$1/b$，更新路径更平滑。
        *   **泛化性**：相比全批量，适度的噪声起到正则化作用，防止过拟合。

---



## 反向传播推导
### MSE Loss，均方误差损失

- MSE Loss就是矩阵的Frobenius范数的平方，直观说明就是矩阵中各个元素平方的和，一般会乘一个1/2。

$$
\begin{array}{l}
\\
L \in R, X \in R ^{B \times D}, Y \in R^{B \times D} \\
B是batch, D是维度, X是预测结果, Y是label, Y的每一行是相等的\\
L = \frac 12 ||X-Y||_F^2  \\
=\frac 12 \sum_{ij}(x_{ij}-y_{ij})^2
\end{array}
$$

- 我们需要的是$L$对$X$的梯度，可以推断出$\frac{\partial L}{\partial X}$的形状和$X$是相同的，就是$B\times D$。现在推导表达式：

$$
\begin{array}{l}
\\
(\frac{\partial L}{\partial X})_{ij} = \frac{\partial }{\partial x_{ij}}\frac 12 \sum_{ij}(x_{ij}-y_{ij})^2 \\
= \frac 12 \frac{\partial}{\partial x_{ij}} (x_{ij}-y_{ij})^2 \\
= x_{ij}-y_{ij} \\
\frac{\partial L}{\partial X} = X- Y\in R^{B\times D}
\end{array}
$$

- 此时已经推导完毕，对于任意定义的Loss，只要它是一个标量，那么它对于输入的$X$的导数的形状就是和$X$相同的，只不过表达式在不同Loss下会不同。

### 矩阵乘法

- 首先明确，我们的目的始终是计算出标量损失L对各个量（可以是矩阵，可以是向量）的导数。对于如下的矩阵乘法：

$$
\begin{array}{l}
\\
Y=X\cdot W \\
Y\in R^{B\times D_2}, X\in R^{B \times D_1}, W\in R^{D_1 \times D_2}\\
\end{array}
$$

- B是batch, D_1是输出维度,D_2是输入维度,X是输入,W是权重矩阵,Y是输出
- 假设我们已经计算得到$\frac{\partial L}{\partial Y}\in R^{B\times D_2}$，为了更新$W$，我们需要$\frac{\partial L}{\partial W}$，为了继续反向传播，我们需要$\frac{\partial L}{\partial X}$。现在推导$\frac{\partial L}{\partial W}$，在此之前，需要先推导$\frac{\partial Y}{\partial W}$。
- 先推断$\frac{\partial Y}{\partial W}$的形状，输出是$B\times D_2$的矩阵, 输入是$D_1\times D_2$的矩阵, 所以导数是一个$B\times D_1 \times D_1 \times D_2$的张量。现在只需要推导出张量中位于ijab位置的元素表达式：$(\frac{\partial Y}{\partial W})_{ijab}=\frac{\partial Y_{ij}}{\partial W_{ab}}$。$Y_{ij}$就是$X$的第i行与$W$的第j列的内积，写成表达式就是：$Y_{ij}=X_{[i, :]}\cdot W_{[:, j]}=\sum_{k} x_{ik}\cdot w_{kj}$，由此可以得到：

$$
(\frac{\partial Y}{\partial W})_{ijab}=\frac{\partial y_{ij}}{\partial w_{ab}}=
\begin{cases}
x_{ia} & j=b\\
0 & j\neq b
\end{cases}
$$

- 不难发现上面表达式在ijab给定时，值是确定的，说明确实可以计算得到任意一个位置的导数。
- 此时我们已经推导出$Y$对于$W$的导数，但是我们的目的始终是计算$L$对各个量的偏导数，所以现在使用链式法则计算$\frac{\partial L}{\partial W}$，用标量推导。设我们已经知道$L$对$Y$的导数$\frac{\partial L}{\partial Y} \in R^{B\times D_2}$，记作$G$：

$$
(\frac{\partial L}{\partial W})_{ab} = \sum_{ij}\frac{\partial L}{\partial y_{ij}}\cdot \frac{\partial y_{ij}}{\partial w_{ab}}=\sum_{ij}G_{ij}\cdot \frac{\partial y_{ij}}{\partial w_{ab}} \\
$$

- 由于$j\neq b$时，$\frac{\partial y_{ij}}{\partial w_{ab}}=0$，这部分直接忽略，由此得到：

$$
\begin{array}{l}
\\
(\frac{\partial L}{\partial W})_{ab} = \sum_{ij}G_{ij}\cdot \frac{\partial y_{ij}}{\partial w_{ab}} \\
只有j=b的时候非零, 把所有j换成b, 然后去掉对j的求和, 以为j=b是一个值, 不需要求和 \\
= \sum_{i}G_{ib}\cdot \frac{\partial y_{ib}}{\partial w_{ab}}  \\
使用之前推导得到的y_{ib}对w_{ab}的表达式, 得到: \\
=\sum_{i}G_{ib}\cdot x_{ia} \\
这等价于G的第b列与X的第a列对应元素相乘然后再相加, 写成内积形式: \\
= X_{[a, :]}^T\cdot G_{[:, b]}\\
(X^T_{[a,:]}表示先转置, 然后再取一行)
\end{array} 
$$

- 此时已经推导得到$(\frac{\partial L}{\partial W})_{ab}=X_{[a,:]}^T\cdot G_{[:, b]}$，可以验证$\frac{\partial L}{\partial W} =  X^T \cdot G \in R^{D_1 \times D_2}$。形状也是对得上的，和$W$的形状相同。
- **由此得到结论**：

$$
\begin{array}{l}
\\
Y=X\cdot W \\
Y\in R^{B\times D_2}, X\in R^{B \times D_1}, W\in R^{D_1 \times D_2} \\
G=\frac{\partial L}{\partial Y} \in R^{B\times D_2} \\
有: \\
\frac{\partial L}{\partial W} =  X^T \cdot G \in R^{D_1 \times D_2}
\end{array}
$$

- 如果我们想要得到$\frac{\partial L}{\partial X}$，可以用同样的方法进行分析，但实际上可以用矩阵转置的形式导出结果。

$$
\begin{array}{l}
\\
Y=X\cdot W \\
转置的性质: \\
Y^T = W^T \cdot X^T \\
利用刚才推导的结果: \\
\frac{\partial L}{\partial X^T} = (W^T)^T\cdot\frac{\partial L}{\partial Y^T} = W\cdot G^T \\
两侧取转置: \\
\frac{\partial L}{\partial X} = G\cdot W^T \\
验证形状: \\
(B, D_1)=(B,D_2)\cdot (D_2\cdot D_1)
\end{array} 
$$

- **由此得到结论**：

$$
\begin{array}{l}
\\
Y=X\cdot W \\
Y\in R^{B\times D_2}, X\in R^{B \times D_1}, W\in R^{D_1 \times D_2} \\
G=\frac{\partial L}{\partial Y} \in R^{B\times D_2} \\
有: \\
\frac{\partial L}{\partial X} =  G \cdot W^T \in R^{B \times D_1}
\end{array}
$$

- 这一套结论就是反向传播中最常用的矩阵乘法的反向传播规则，有以下直观的发现。
  - 为了计算$L$对$W$的导数，我们需要在前向传播的时候保存$X^T$。直观理解就是每个线性层的输出都需要保存一份。
  - 虽然推导中需要计算$Y=X\cdot W$中$Y$对$X$的导数，而且发现这个导数是一个四维张量，无论是计算还是存储都有很大代价。但是实际计算中，只需要分别进行一次矩阵乘法就可以得到需要的$L$对$W$和$L$对$X$的导数，计算非常简洁，实现起来也十分容易。

### 矩阵加法，bias

- **矩阵加法**：仍然先明确，我们的目的是计算标量损失L对各个量的导数。那么对于如下的矩阵加法：

$$
\begin{array}{l}
\\
Y = X + B  \\
Y, X,B \in R^{B\times D}
\end{array}
$$

- B是batch，D是维度。假设我们已经计算得到$\frac{\partial L}{\partial Y}\in R^{B\times D_2}$，现在推导$\frac{\partial L}{\partial B}$，在此之前，需要先推导$\frac{\partial Y}{\partial B}$。
- 先推断$\frac{\partial Y}{\partial B}$的形状，输出是$B\times D$的矩阵，输入是$B\times D$的矩阵，所以导数是一个$B\times D \times B \times D$的张量，现在只需要推导出张量中位于ijab位置的元素表达式：$(\frac{\partial Y}{\partial B})_{ijab}=\frac{\partial Y_{ij}}{\partial B_{ab}}$，结果非常直观：

$$
(\frac{\partial Y}{\partial B})_{ijab}=\frac{\partial y_{ij}}{\partial b_{ab}}=
\begin{cases}
1 & i=a ,j=b\\
0 & \text{others}
\end{cases}
$$

- 此时我们已经推导出$Y$对于$B$的导数，现在使用链式法则计算$\frac{\partial L}{\partial B}$，用标量推导，设我们已经知道$L$对$Y$的导数$\frac{\partial L}{\partial Y}\in R^{B\times D}$，记作$G$：

$$
(\frac{\partial L}{\partial B})_{ab} = \sum_{ij}\frac{\partial L}{\partial y_{ij}}\cdot \frac{\partial y_{ij}}{\partial b_{ab}}=\sum_{ij}G_{ij}\cdot \frac{\partial y_{ij}}{\partial b_{ab}} \\
$$

- 由于只有$i=a,j=b$时导数非零，所以得到：

$$
\begin{array}{l}
\\
(\frac{\partial L}{\partial B})_{ab}=G_{ab}\cdot 1=G_{ab} \\
\frac{\partial L}{\partial B}=G \in R^{B\times D}
\end{array}
$$

- 此时已经推导完毕，可以发现对于矩阵加法，L对各个矩阵的导数就是L对各个矩阵的和的导数。**由此得到结论**：

$$
\begin{array}{l}
\\
Y=X+B \\
Y,X,B \in R^{B\times D} \\
G = \frac{\partial L}{\partial Y} \in R^{B\times D} \\
有: \\
\frac{\partial L}{\partial B}= G \in R^{B \times D}
\end{array}
$$

- 对于$\frac{\partial L}{\partial X}$，结论显然是一样的。
- **加bias**：虽然添加bias这个操作完全可以合并到矩阵乘法中，不过这种操作非常常见，这里还是推导一下。对于$Y=X+b$，在数学上一定是要求$X$和$b$形状相同的；不过在torch中，因为广播机制，实际上计算的是$Y=X+\text{ones}(B, 1) \cdot b$，其中$Y,X \in R^{B \times D}, b\in R^{1 \times D}$。因为我们已经推导过矩阵加法的导数，所以现在只需要推导如下内容：

$$
\begin{array}{l}
\\
b \in R^{1 \times D} \\
B=\text{ones}(B, 1)\cdot b \\
x是b的索引 \\
(\frac{\partial B}{\partial b})_{ijx}= 
\begin{cases}
1 & j=x \\
0 & \text{others}
\end{cases}
\end{array}
$$

- 这个推导和之前是类似的，注意此时$\frac{\partial B}{\partial b}$的形状应该是$B \times D \times B$，只有当$B_{ij}$的列索引和$b$的索引相同的时候，导数才是1，否则都是0。
- 设我们已知$\frac{\partial L}{\partial Y} \in R^{B \times D}$，记作$G$，根据之前对矩阵加法导数的推导，对于$Y=X+B$，有$\frac{\partial Y}{\partial B}=G$，现在使用链式法则推导$\frac{\partial L}{\partial b}$：

$$
\begin{array}{l}
\\
(\frac{\partial L}{\partial b})_x = \sum_{ij}G_{ij}\frac{\partial B_{ij}}{\partial b_x} \\
= \sum_i G_{ix}\frac{\partial B_{ix}}{\partial b_x} \\
= \sum _i G_{ix} \cdot 1\\
\frac{\partial L}{\partial b} = \text{ones(1, B)}\cdot G \in R^{1 \times D}
\end{array}
$$

- 由此得到结论：

$$
\begin{array}{l}
\\
Y=X+\text{ones}(B, 1)\cdot b=X+B \\
Y, X, B \in R^{B \times D}, b \in R^{1 \times D}\\
G=\frac{\partial L}{\partial Y} \in R^{B \times D} \\
有: \\
\frac{\partial L}{\partial b}=\sum_{i=1}^B G_{i, :}=\text{ones}(1, B)\cdot G \in R^{1 \times D}
\end{array}
$$

- 直观理解就是把L对Y的导数沿着batch维度求和，然后就得到了L对b的梯度。

### 激活函数，element-wise的操作

- 激活函数普遍都是element-wise的操作，所以自然有如下的推导

$$
\begin{array}{l}
\\
Y=\delta(X)\\
Y,X\in R^{B \times D} \\
先推导\frac{\partial Y}{\partial X}: \\
\frac{\partial Y}{\partial X} \in R^{B\times D \times B \times D} \\
\frac{\partial y_{ij}}{\partial x_{ab}} = \begin{cases}
\delta^{'}(x_{ab}) & i=a,j=b \\
0 & \text{others}
\end{cases} \\
继续推导\frac{\partial L}{\partial X}:
\\
已知\frac{\partial L}{\partial Y}=G\in R^{B \times D}\\
有(\frac{\partial L}{\partial X})_{ab}= \sum_{ij}\frac{\partial L}{\partial y_{ij}}\cdot \frac{\partial y_{ij}}{\partial x_{ab}} \\
=\frac{\partial L}{\partial y_{ab}}\cdot \frac{\partial y_{ab}}{\partial x_{ab}} \\
=\frac{\partial L}{\partial y_{ab}}\cdot \delta^{'}(x_{ab}) \\
得到: \\
\frac{\partial L}{\partial X} = G \odot \delta^{'}(X) \\
这里的\odot 表示哈达玛积, 就是对应元素相乘\\
\end{array}
$$

- 直观理解就是把激活函数的导数作用在输入矩阵的每个元素上，然后将这个矩阵与L对激活值得导数计算哈达玛积，得到的就是L对输入矩阵的导数。
- 下面是常见激活函数的导数的总结：

| **操作**       | **前向计算 Z=f(Y)**                     | **梯度 ∂Y∂L 的通用公式**               |
| -------------- | --------------------------------------- | -------------------------------------- |
| ReLU           | $Z = \max(0, Y)$                        | $G_Y = G_Z \odot \mathbf{1}_{\{Y>0\}}$ |
| Sigmoid        | $Z = \frac{1}{1+e^{-Y}}$                | $G_Y = G_Z \odot Z \odot (1-Z)$        |
| Tanh           | $Z = \frac{e^Y - e^{-Y}}{e^Y + e^{-Y}}$ | $G_Y = G_Z \odot (1-Z^2)$              |
| 幂函数 (Power) | $Z = Y^k$                               | $G_Y = G_Z \odot k \cdot Y^{k-1}$      |
| 对数 (Log)     | $Z = \ln(Y)$                            | $G_Y = G_Z \odot \frac{1}{Y}$          |

- 实际上哈达玛积本身也是一个element-wise的操作：

$$
\begin{array}{l}
\\
Y=P\odot Q \\
Y,P,Q\in R^{B\times D} \\
先推导 \frac{\partial Y}{\partial P} \in R^{B\times D \times B \times D} \\
\frac{\partial y_{ij}}{\partial p_{ab}} = 
\begin{cases}
q_{ab} & a=i,b=j \\
0 & \text{others}
\end{cases} \\
继续推导\frac{\partial L}{\partial P}\\
设\frac{\partial L}{\partial Y} =G \in R^{B \times D} \\
\frac{\partial L}{\partial p_{ab}}=\sum_{ij}\frac{\partial L}{\partial y_{ij}}\cdot \frac{\partial y_{ij}}{\partial p_{ab}} \\
=G_{ab}\cdot q_{ab} \\
\frac{\partial L}{\partial P} = G \odot Q
\end{array}
$$

### 降维操作 

- **对整个向量求和**：$y=\sum_{ij} X_{ij}，y\in R, X\in R^{B\times D}$，求$\frac{\partial y}{\partial X}$以及$\frac{\partial L}{\partial X}$。思路自然是写出$y$对X每个元素的导数，推导如下：

$$
\begin{array}{l}
\\
推断形状: 
\frac{\partial y}{\partial X} \in R ^{B \times D} \\
\frac{y}{\partial x_{ij}} = 1 \\
\frac{\partial y}{\partial X} = \text{ones}(B, D) \\
设已知\frac{\partial L}{\partial y} = g \in R\\
推导\frac{\partial L}{\partial X}: \\
\frac{\partial L}{\partial x_{ij}} = \frac{\partial L}{\partial y}\cdot \frac{\partial y}{\partial x_{ij}} = \frac{\partial L}{\partial y}=g \\
得到\frac{\partial L}{\partial X} = g \cdot \text{ones}(B, D)\in R^{ B \times D}
\end{array}
$$

- 对于mean操作，显然有$\frac{\partial Y}{\partial X}=\frac{1}{B \times D} \cdot \text{ones} (B, D)$，自然得到$\frac{\partial L}{\partial X} = g \cdot \frac{1}{B\times D} \cdot \text{ones}(B, D)\in R^{ B \times D}$。

- **maxpooling**：推导如下：

$$
\begin{array}{l}
\\
y=\max(X), y\in R, X\in R^{B \times D} \\
推断形状: \\
\frac{\partial y}{\partial X} \in R ^{B \times D} \\
\frac{\partial y}{\partial x_{ij}} = \begin{cases}
1 & i,j=\arg \max X\\
0 & \text{others}
\end{cases} \\
设已知\frac{\partial L}{\partial y}=g \in R \\
\frac{\partial L}{\partial x_{ij}} = \begin{cases}
g \cdot 1 & i,j=\arg \max X \\
0 & \text{others}
\end{cases} \\
\frac{\partial L}{\partial X}\in R^{B \times D}
\end{array}
$$

- 可以发现，maxpooling操作实际上就是只保留了最大的输入元素的梯度，其余元素的梯度都是零。

### softmax

- **softmax本身的导数**：
- softmax本身不是一个降维操作，但它和cross-entropy结合之后就构成了一个非常常见的损失函数。先推导softmax本身的导数。

$$
\begin{array}{l}
\\
设输入的样本为: \\
z = (z_1, z_2, \dots, z_C) \in R^{1 \times C} \\
softmax的定义为: \\
p_i = \frac{e^{z_i}}{\sum_{k=1}^C e^{z_k}}, \quad p \in R^{1 \times C} \\
计算\frac{\partial p}{\partial z},先推断形状: \\
\frac{\partial p}{\partial z} \in R^{C \times C} \\
当i=j时 \\
\frac{\partial p_i}{\partial z_i}= \frac{e^{z_i}(\sum_k e^{z_k}) - e^{z_i}e^{z_i}}{(\sum_k e^{z_k})^2}=\frac{e^{z_i}}{\sum_k e^{z_k}}- \frac{e^{z_i}\cdot e^{z_i}}{(\sum_k e^{z_k})^2} \\
= p_i - p_i^2 \\
= p_i(1 - p_i) \\
当i\neq j时 \\
\frac{\partial p_i}{\partial z_j}
= -\frac{e^{z_i}e^{z_j}}{(\sum_k e^{z_k})^2}
= -p_i p_j \\
合并写为: \\
\frac{\partial p_i}{\partial z_j}
= p_i(\delta_{ij} - p_j) \\
其中\delta_{ij}仅在i=j时为1, 否则为0\\
可以写成矩阵形式: \\
\frac{\partial p}{\partial z}
= \mathrm{diag}(p) - p p^\top
\in R^{C \times C}
\end{array}
$$

- 现在假设已知$\frac{\partial L}{\partial p}=G \in R^{1 \times C}$，继续推导$\frac{\partial L}{\partial z}$：

$$
\begin{array}{l}
\\
推断形状: \\
\frac{\partial L}{\partial z}\in R ^{1 \times C} \\
\frac{\partial L}{\partial z_a}=\sum_i \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_a} =\sum_i G_i \cdot p_i (\delta_{ia}-p_a) \\
=\sum_i G_i \cdot p_i \cdot \delta_{ia}- \sum_i G_i \cdot p_i \cdot p_a \\
= G_a \cdot p_a - p_a\sum _i G_i \cdot p_i \\
写成矩阵形式(形式不唯一且不重要): \\
\frac{\partial L}{\partial z} = G \odot p - (G\cdot p^T)\cdot p \\
其中G\cdot p^T是一个标量
\end{array}
$$

- **cross-entropy的导数**：
- 现在推导L对cross-entropy的输入的导数$\frac{\partial L}{\partial p}$：

$$
\begin{array}{l}
\\
cross-entropy定义: \\
L = -\sum_{i=1}^C y_i \log p_i,\quad L \in R \\
其中y_i是标签, p_i是softmax的输出 \\
推断形状: \\
\frac{\partial L}{\partial p} \in R ^{1\times C} \\
\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}\\
这里缺乏一个严谨的符号来写出矩阵形式的结果, 用./表示对应元素相除: \\
\frac{\partial L}{\partial p} = -y./p \in R^{1 \times C}
\end{array}
$$

- **softmax接cross-entropy的导数**：
- 在已经前述推导的基础上，此时可以推导L对softmax的输入的导数$\frac{\partial L}{\partial z}$：

$$
\begin{array}{l}
\\
已知 p=softmax(z)\in R^{1 \times C}, \quad p, z \in R^{1 \times C} \\
L = -\sum_{i=1}^C y_i \log p_i,\quad L \in R, y\in R^{1 \times C} \\
根据前面推导, 有: \\
\frac{\partial L}{\partial z_a}
= G_a \cdot p_a - p_a\sum _i G_i \cdot p_i \\
也有: \\
G_i = \frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i} \\
两者结合, 得到: \\
\frac{\partial L}{\partial z_a} = G_a \cdot p_a - p_a\sum _i G_i \cdot p_i \\
= -\frac{y_a}{p_a}\cdot p_a + p_a\sum_i \frac{y_i}{p_i}\cdot p_i \\
=-y_a + p_a\sum_i y_i \\
如果y是one-hot矩阵, 则y中只有一个元素是1, 其余都是0, 自然有\sum_iy_i=1,得到: \\
原式=-y_a + p_a =p_a-y_a\\
写成矩阵形式: \\
\frac{\partial L}{\partial z} = p-y\in R^{1 \times C}
\end{array}
$$

- 从结果中可以看到，softmax后面接cross-entropy之后，损失L对softmax的输入的梯度的表达式是非常简洁的。这也是为什么在torch等框架中，softmax和cross-entropy通常被融合成一个算子。
- **log-softmax的说明**：
- 在torch中，为了数值稳定性，实际计算的是log-softmax。它是为了解决如下的问题的。
- 如果没有log-softmax操作，用softmax的定义计算，如果某个$z_i$特别大，那么在计算$p_i$的时候，分子分母都很大会直接溢出。如果某个$z_i$特别小，比如是-1000，那么计算出来的$p_i$会直接数值下溢变成0，后续计算$\log p_i$的时候会有问题，同时也无法区分出-1000和-10000这种logits的差异（因为概率都下溢成0了，没区别了）。注意到在softmax之后计算cross-entropy时，需要的不是$p_i$，而是$\log p_i$，所以可以考虑直接计算$\log p_i$，而不是先计算$p_i$再取log。至于梯度计算时需要的$p_i$，只需要对先计算出的$\log p_i$再取一次指数就可以了。
- 下面是log-softmax的forward说明：

$$
\begin{array}{l}
\\
\mathrm{log\_softmax}(z_i) = \log(p_i)=
\log\frac{e^{z_i}}{\sum_{j} e^{z_j}} \\
= \log(e^{z_i}) -
\log\left(\sum_{j} e^{z_j}\right) \\
=\log (e^{z_i})-\log (e^m \sum _j e^{z_j - m}) \\
=z^i - m - \log (\sum e^{z_j -m}) \\
其中m=\max_j z_j \\
此时保证了 z_j - m \le 0 \Rightarrow e^{z_j -m }\in (0, 1] \\
由此避免了e^{z_j}过大导致的数值上溢 \\
同时没有先计算p_i再计算\log(p_i), 避免了接近0的p_i导致的\log(p_i)的数值下溢问题
\end{array}
$$

- 使用log-softmax直接计算出$\log p_i$，然后代入cross-entropy，在forward上和原来完全等价，梯度自然也是完全相同的。

### 卷积

- **经典推导**：卷积导数的经典推导参考:<https://zhuanlan.zhihu.com/p/640697443>，类似的推导的思路基本上都是把卷积写成标量求和的形式，然后得到结果。
- **利用之前推导的结果**：实际上卷积操作涉及的所有运算的导数我们都推导过，对于每个位置的卷积，实际上就是$y_{ij}=\text{sum}(X_{[i+kw,j+kh]} \odot K)$，也就是卷积核与输入矩阵的一部分计算哈达玛积，然后求和。在已知$\frac{\partial L}{\partial Y}$的情况下，用之前的求导结果可以算出来$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y_{ij}}\cdot \frac{\partial y_{ij}}{\partial K}$和$\frac{\partial L}{\partial y_{ij}}\cdot \frac{\partial y_{ij}}{\partial X_{[i+kw,j+kh]}}$。卷积核的移动操作会把导数积累到$\frac{\partial L}{\partial K}$和$\frac{\partial L}{\partial X}$上，即便没有显示地推导出导数的形式，也完全可以计算卷积的导数。
- 将一些常用操作的导数手动推导出来的意义是可以合并中间结果，有望减少中间计算和中间结果的存储。
- **实际的卷积**：实际的卷积计算无论forward还是backward都是用矩阵乘法完成的，并不存在用卷积核在矩阵上移动计算的步骤。考虑每个位置的计算：$y_{ij}=\text{sum}(X_{[i+kw,j+kh]} \odot K)$，如果将$X_{[i+kw,j+kh]}$和$K$都展平成向量，那么这个计算可以表达成两个向量的内积形式。将$X$每次计算的切片部分都展平成向量然后按照列拼起来，那么整个卷积运算就可以表示成一个矩阵和向量的乘法计算。下面是一个例子：
- 设输入样本为：
$$
X =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
\in \mathbb{R}^{1\times1\times4\times4}
$$
- 设卷积核为：
$$
W =
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}
\in \mathbb{R}^{1\times1\times3\times3}
$$
- 设padding=0，stride=1，输出的特征图尺寸为$4-3+1=2$，直接计算卷积的结果如下（左上角为例）：
$$
Y_{1,1} =
\begin{bmatrix}
1 & 2 & 3 \\
5 & 6 & 7 \\
9 & 10 & 11
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}
$$
- 最终的结果为：
$$
Y =
\begin{bmatrix}
-6 & -6 \\
-6 & -6
\end{bmatrix}
$$
- 下面是转换成矩阵乘法的计算过程。按照行拼接的顺序，将各个位置的$3 \times 3$patch拼成列向量：
$$
X_{col} =
\begin{bmatrix}
1 & 2 & 5 & 6 \\
2 & 3 & 6 & 7 \\
3 & 4 & 7 & 8 \\
5 & 6 & 9 & 10 \\
6 & 7 & 10 & 11 \\
7 & 8 & 11 & 12 \\
9 & 10 & 13 & 14 \\
10 & 11 & 14 & 15 \\
11 & 12 & 15 & 16
\end{bmatrix}
\in \mathbb{R}^{9\times4}
$$
- 将卷积核按照行拼接的顺序展平成行向量：
$$
W_{row} =
\begin{bmatrix}
1 & 0 & -1 & 1 & 0 & -1 & 1 & 0 & -1
\end{bmatrix}
\in \mathbb{R}^{1\times9}
$$
- 用矩阵乘法实现卷积：
$$
\begin{array}{l}
\\
Y_{col} = W_{row} \cdot X_{col}  \\
= [-6,-6,-6,-6] \\
\end{array}
$$
- 将结果按照行拆分reshape回正常的输出形状：
$$
Y =
\begin{bmatrix}
-6 & -6 \\
-6 & -6
\end{bmatrix}
\in \mathbb{R}^{1\times1\times2\times2}
$$
- **实际的卷积的导数**：当卷积计算被转化为矩阵乘法之后，我们就可以找到一种新的推导卷积导数的方法。矩阵乘法的导数在前面已经推导过，所以在与卷积等效的矩阵乘法$Y_{col} = W_{row} \cdot X_{col}$，中，自然可以计算出$\frac{\partial L}{\partial W_{row}}$和$\frac{\partial L}{\partial X_{col}}$。对于卷积核的导数$\frac{\partial L}{\partial W_{row}}$来说，只要reshape回卷积核的形状就完成了计算。对于输入特征图对应的矩阵的导数$\frac{\partial L}{\partial X_{col}}$，需要将对应位置的梯度累加到原特征图对应的位置上。
	- 用这种方式在直觉上可以验证为什么卷积的导数是卷积核转置之后在$\frac{\partial L}{\partial Y}$上进行卷积运算。
- **工程实现的思路**：实际用GPGPU实现卷积的forward和backward时，思路是每个threadblock负责输出结果的一块，每个threadblock把kernel和需要的输入矩阵读进shared memory，之后在线从shared memory中读取元素得到patch展平的向量，然后计算结果并写入输出的对应位置。在整个过程中不需要显示地构造patch展平之后拼接的矩阵。

---


# TODO
## Future Work
1.  **更先进的优化器**
2.  **二阶优化方法**：探索利用损失函数的二阶导数（海森矩阵或其近似）信息的优化方法，如自然梯度下降、KFAC等，以期获得更优的收敛特性，尽管其计算和存储成本通常更高。
3.  **优化理论的新理解**：深入研究为什么带噪声的SGD（以及小批量SGD）往往比全批量梯度下降找到的解泛化能力更好，探索损失函数几何景观（如平坦极小值）与泛化性能之间的联系。
4.  **大规模分布式训练**：针对超大模型和数据集，研究高效的分布式优化算法，如数据并行、模型并行、流水线并行以及它们的混合策略，以解决单机资源瓶颈。
5.  **训练稳定性与加速**：研究梯度裁剪、学习率预热、动态批次大小等技术，以稳定超大规模模型的训练过程，并进一步提升训练效率。
6.  **元学习与自动化优化**：模型如何学会如何优化自身（学习优化器），或使用自动化机器学习（AutoML）技术来动态调整优化超参数（如学习率调度）。


## Practice in LLM
### LLM训练的本质
1. **LLM简介**
大型语言模型（Large Language Model, LLM）是一种基于Transformer架构构建的、参数量极其庞大（通常为数十亿至数万亿）的深度学习模型。其核心目标是通过在海量无标注文本数据上进行自监督预训练，学习人类语言的通用表示、语法规则、世界知识以及逻辑推理能力。LLM通过预测文本序列中的下一个词（或词元）这一基础任务，掌握了语言的生成与理解能力，并能够通过指令微调、对齐等技术，泛化到多种下游任务，展现出强大的通用人工智能潜力。

2. **问题定义**
    *   **目标**：在给定海量文本数据集 $ \mathcal{D} $ 上，寻找一组最优的模型参数 $ \theta $，最小化基于自监督任务（如语言建模）的损失函数 $ L(\theta; \mathcal{D}) $，使模型能够准确预测或生成符合语言规律和上下文逻辑的文本。
    *   **挑战**：
        1.  **前所未有的规模**：参数量（$ \|\theta\| $）达到 $10^9 \sim 10^{12}$ 级别，导致优化空间维度极高，对梯度计算（反向传播）和参数更新的存储、通信带来极限压力。
        2.  **海量数据**：训练数据可达万亿词元量级，遍历一次（一个epoch）的计算成本极高，要求优化算法在极少的数据遍历次数内高效收敛。
        3.  **训练不稳定性**：在超大规模模型训练中，梯度爆炸/消失、损失值突变（loss spike）、数值溢出等问题更为常见且危害巨大。
        4.  **泛化与记忆的平衡**：模型容量极大，极易完全记忆训练数据而导致过拟合，需要在优化过程中引入有效的隐式或显式正则化，促使模型学习可泛化的模式而非噪声。
        5.  **计算资源瓶颈**：单卡内存无法容纳整个模型和优化器状态，必须依赖复杂的分布式并行训练策略（如数据并行、模型并行、流水线并行），这使得优化算法的设计与系统工程深度耦合。

3. **核心方法论的延续与演进**
   LLM的训练本质上仍然是遵循“反向传播提供梯度，梯度下降指导更新”的范式，但面对上述挑战，其具体实现发生了深刻演进：
    *   **反向传播（Backpropagation）**：**基础引擎依然核心**。其高效计算梯度的能力是训练任何深度网络的基石。在LLM中，反向传播的计算图因Transformer的复杂结构（多头注意力、前馈网络、残差连接、层归一化）而变得巨大，但其链式法则的本质未变。核心挑战在于如何**分布式地、高效地**完成跨数千张GPU的反向传播计算。
    *   **梯度下降（Gradient Descent）**：**导航仪需要全面升级**。朴素的SGD已无法应对LLM的训练。其演进体现在：
        1.  **优化器升级**：采用**自适应学习率优化器（如Adam, AdamW）**。它们为每个参数维护独立的学习率，能更平稳地处理不同参数尺度和稀疏梯度，是训练稳定的关键。
        2.  **动态调度**：引入复杂的学习率调度（如余弦退火、线性预热），在训练初期稳定，后期精细调优。
        3.  **稳定化技术**：梯度裁剪（Gradient Clipping）成为标配，防止异常梯度导致训练崩溃。
        4.  **并行化策略**：梯度下降的更新步骤必须与分布式并行策略协同设计。例如，在数据并行中，需要跨设备同步梯度；在模型并行中，参数更新本身也是分布式的。

---

### 模型损失函数 && 反向传播
在LLM的预训练阶段，最主流的损失函数是**因果语言建模（Causal Language Modeling）的交叉熵损失**，也称**下一个词元预测损失**。

给定一个文本序列 $ \mathbf{x} = (x_1, x_2, ..., x_T) $，模型将其编码为词元序列。在每一步 $ t $，模型基于之前的所有词元 $ x_{<t} $ 预测下一个词元 $ x_t $ 的概率分布 $ P_\theta(x_t | x_{<t}) $。损失函数是每一步预测的负对数似然之和的平均：
$$
L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})
$$
其核心是衡量模型预测的分布与真实“one-hot”分布（真实的下一个词元）之间的差异。

**LLM反向传播的特点**：
1.  **序列化的计算图**：由于自回归特性，第 $ t $ 步的计算依赖于前 $ t-1 $ 步的所有中间激活值。这导致了巨大的**内存开销（激活值存储）**，成为训练的主要瓶颈之一。技术如**激活重计算（Activation Checkpointing）** 被广泛使用，即在前向传播时只存储部分层的激活值，在反向传播时临时重算其余部分，以时间换空间。
2.  **注意力机制的梯度流**：Transformer中的自注意力机制引入了所有词元对之间的交互。反向传播时需要计算损失对每个注意力权重和中间表示的梯度，这部分计算量巨大，但也是模型学习长程依赖的关键。
3.  **分布式反向传播**：在模型并行（如Tensor Parallelism）下，单个层的计算被分割到多个设备上。反向传播需要精心设计通信操作，以确保跨设备的梯度能够被正确聚合和同步，这通常由深度学习框架（如PyTorch的`FSDP`、Megatron-LM）在底层自动管理。

---

### 梯度下降更新权重
LLM训练中，权重更新不再是小规模网络中的简单步骤，而是一个集成了多种技术的复杂过程。

1.  **优化器：AdamW**

2.  **关键训练技术**：
    *   **混合精度训练**：使用FP16/BF16进行前向和反向传播以加速计算、节省内存，同时保留FP32的权重主副本用于更新，以保持数值稳定性。
    *   **梯度裁剪**：在反向传播后、优化器更新前，将梯度向量的范数限制在一个阈值内，防止训练因梯度爆炸而失效。
    *   **学习率调度**：采用“预热+衰减”策略。训练初期进行数千步的**线性预热**，让学习率从0缓慢增至峰值，使优化初期更稳定。之后采用**余弦衰减**等方式缓慢降低学习率，帮助模型收敛到更优的极小点。

3.  **批大小（Batch Size）的宏观策略**：
    LLM训练通常使用极大的**全局批大小**（Global Batch Size），可能是数百万个词元。这是通过**数据并行**将大批量分割到成千上万个GPU上实现的。增大批大小可以更准确地估计梯度方向，从而允许使用更高的学习率，理论上能加快收敛。但批大小与学习率、模型大小之间存在复杂的权衡关系，需要根据经验法则（如“√倍批大小，√倍学习率”）和大量实验来确定最优配置。

---
