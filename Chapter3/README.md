# Transformer
## 一、前言
Transformer 是由 Google 团队在 2017 年发表的论文《Attention Is All You Need》中提出的一种神经网络架构，其核心创新在于完全基于注意力机制（Attention Mechanism），摆脱了传统循环神经网络（RNN）、长短期记忆网络（LSTM）等对序列顺序的依赖，实现了并行化训练，极大地提升了模型的训练效率和性能。该架构已成为自然语言处理（NLP）领域的基础模型，广泛应用于机器翻译、文本生成、情感分析等任务，同时也在计算机视觉（CV）等其他领域展现出强大的潜力。

## 二、Transformer 整体结构
Transformer 的整体结构由 Encoder（编码器）和 Decoder（解码器）两大部分组成，中间通过 Cross-Attention（交叉注意力）模块实现连接，具体结构特点如下：
1. **核心组成**：左半部分为 Encoder 堆叠结构，右半部分为 Decoder 堆叠结构，二者通过 Cross-Attention 传递编码信息。
2. **输入端**：输入数据需经过 Word Embedding（词嵌入）和 Positional Embedding（位置编码）处理，将离散的单词转换为连续的向量，并融入单词在序列中的位置信息。
3. **中间层**：
    - Encoder 由 N 个相同的 Encoder Block 堆叠而成，每个 Block 包含 Multi-Head Attention（多头注意力）和 Feed Forward（前馈全连接）两个核心子层，且每个子层前后均设有 Add & Norm（残差连接和层归一化）操作。
    - Decoder 同样由 N 个相同的 Decoder Block 堆叠而成，每个 Block 包含 Masked Multi-Head Attention（掩码多头注意力）、Cross-Attention 和 Feed Forward 三个核心子层，各子层前后也均设有 Add & Norm 操作。
4. **输出端**：Decoder 的输出经过 Linear（线性变换）将向量维度映射到词汇表大小，再通过 Softmax 函数转换为概率分布，最终输出每个位置对应的预测单词概率。

## 三、Transformer 的输入
Transformer 的输入处理包含两个关键步骤：Word Embedding 和 Positional Embedding，最终将二者相加得到模型的输入向量。

### （一）Word Embedding（词嵌入）
Word Embedding 的核心作用是将离散的单词符号转换为连续的低维向量表示，使计算机能够理解单词的语义信息。例如，对于输入句子“Je suis étudiant”（法语，意为“我是学生”），通过 Word Embedding 可将每个单词“Je”“suis”“étudiant”分别映射为固定维度的向量 X1、X2、X3。这些向量会捕捉单词之间的语义关联（如“étudiant”与“student”的向量具有较高相似度）。

### （二）Positional Embedding（位置编码）
由于 Transformer 不依赖循环结构，无法像 RNN 那样自然捕捉序列的顺序信息，因此需要通过 Positional Embedding 手动注入位置信息。位置编码的计算公式如下：
- 对于位置 pos 和维度 i，若 i 为偶数：$PE(pos, i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
- 对于位置 pos 和维度 i，若 i 为奇数：$PE(pos, i) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
其中，$d_{model}$ 为词嵌入向量的维度。位置编码向量与词嵌入向量维度相同，二者通过元素相加的方式融合，最终得到包含语义和位置信息的输入向量。

## 四、Self-Attention（自注意力机制）
Self-Attention 是 Transformer 的核心部件，Encoder 和 Decoder 均以其为基础构建（Decoder 中为 Masked 版本）。Multi-Head Attention 则是由多个独立的 Self-Attention 并行计算后拼接而成，能够从多个视角捕捉序列内的依赖关系。

### （一）Self-Attention 的计算步骤
以输入序列“Thinking Machines”（对应的词嵌入向量为 X1、X2）为例，Self-Attention 的计算过程如下：
1. **步骤 1：生成 Query、Key、Value 向量**
对每个输入向量 X（X1、X2）进行线性变换，分别得到三个向量：
    - Query（查询向量，Q）：用于查询其他位置的信息。
    - Key（键向量，K）：用于与其他位置的 Query 向量计算相似度。
    - Value（值向量，V）：存储当前位置的核心信息，将根据相似度权重进行聚合。
例如，X1 经线性变换得到 Q1、K1、V1，X2 得到 Q2、K2、V2。

2. **步骤 2：计算 Attention Score（注意力分数）**
注意力分数用于衡量当前位置 Query 与其他位置 Key 的关联程度，通过 Query 与 Key 的点积计算得到。例如，Q1 与 K1 的点积为 112，Q1 与 K2 的点积为 96，这些分数反映了“Thinking”与自身及“Machines”的关联强度。

3. **步骤 3：归一化注意力分数**
为了避免点积结果过大导致 Softmax 函数饱和，将注意力分数除以 $\sqrt{d_k}$（$d_k$ 为 Key 向量的维度，论文中取 8），对分数进行缩放。例如，112/8=14，96/8=12。

4. **步骤 4：Softmax 归一化**
对缩放后的注意力分数应用 Softmax 函数，将其转换为 0-1 之间的概率分布，且所有概率之和为 1。例如，14 和 12 经 Softmax 后得到 0.88 和 0.12，该分布表示当前位置对其他位置信息的关注权重。

5. **步骤 5：分数与 Value 向量相乘**
将 Softmax 得到的权重与对应位置的 Value 向量进行元素相乘，得到加权后的 Value 向量。例如，0.88×V1、0.12×V2。

6. **步骤 6：聚合加权 Value 向量**
将所有加权后的 Value 向量相加，得到当前位置的 Self-Attention 输出向量。例如，Z1=0.88×V1 + 0.12×V2，Z2 同理。该输出向量融合了序列中所有位置的信息，并根据关联强度分配了不同权重。

### （二）Self-Attention 矩阵计算
从矩阵角度，Self-Attention 的计算可表示为：
$Attention(Q, K, V) = Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
其中，Q、K、V 分别为 Query、Key、Value 的矩阵形式（每行对应一个位置的向量），$QK^T$ 为注意力分数矩阵，经缩放、Softmax 归一化后与 V 矩阵相乘，得到最终的注意力输出矩阵。

### （三）Multi-Head Attention（多头注意力机制）
Multi-Head Attention 通过将 Q、K、V 向量分别线性投影到 h 个不同的子空间（h 为头数），在每个子空间中独立计算 Self-Attention，最后将 h 个 Self-Attention 的输出向量拼接起来，再通过一次线性变换得到最终结果。其公式为：
$MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h)W^O$
其中，$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$，$W_i^Q$、$W_i^K$、$W_i^V$ 分别为第 i 个头的投影矩阵，$W^O$ 为拼接后的线性变换矩阵。

多头注意力的优势在于能够同时捕捉不同类型的依赖关系（如语法依赖、语义依赖），每个头专注于一个特定的视角，从而提升模型的表达能力。

## 五、Encoder 结构
Encoder 由 N 个 Encoder Block 堆叠而成（论文中 N=6），每个 Block 的结构统一，核心是实现对输入序列的编码，提取上下文信息。

### （一）Encoder Block 的组成
每个 Encoder Block 包含以下四个部分，按顺序执行：
1. **Multi-Head Attention 层**：对输入序列进行多头自注意力计算，捕捉序列内的上下文依赖关系。
2. **Add & Norm 层（残差连接 + 层归一化）**：
    - Add：残差连接，将 Multi-Head Attention 层的输入与输出相加，即 $X + MultiHeadAttention(X)$，用于缓解深度网络训练中的梯度消失问题。
    - Norm：层归一化（Layer Normalization），将相加后的向量进行归一化处理，使每个神经元的输入分布保持稳定，加速网络收敛。归一化公式为：$LayerNorm(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$，其中 $\mu$ 为均值，$\sigma^2$ 为方差，$\gamma$ 和 $\beta$ 为可学习参数，$\epsilon$ 为防止分母为 0 的微小值。
3. **Feed Forward 层（前馈全连接层）**：一个两层的全连接网络，对每个位置的向量进行独立的非线性变换，公式为：$FFN(X) = max(0, XW_1 + b_1)W_2 + b_2$。其中，第一层使用 ReLU 激活函数（$max(0, \cdot)$），将负值归零，引入非线性；第二层为线性变换，映射回原向量维度。
4. **Add & Norm 层**：再次执行残差连接（$X + FFN(X)$）和层归一化，输出当前 Encoder Block 的结果，作为下一个 Block 的输入。

### （二）Encoder 的整体工作流程
1. 第一个 Encoder Block 的输入为经过 Word Embedding 和 Positional Embedding 融合后的向量矩阵 X（每行对应一个单词的向量）。
2. 每个 Encoder Block 对输入矩阵进行处理，逐步提取更高级的上下文特征。
3. 最后一个 Encoder Block 输出的矩阵 C 为整个输入序列的编码信息矩阵，该矩阵将作为 Decoder 中 Cross-Attention 层的 Key 和 Value 输入，为解码过程提供上下文支持。

## 六、Decoder 结构
Decoder 由 N 个 Decoder Block 堆叠而成（论文中 N=6），核心是基于 Encoder 的编码信息和已生成的序列前缀，预测下一个单词。

### （一）Decoder Block 的组成
每个 Decoder Block 包含以下五个部分，按顺序执行：
1. **Masked Multi-Head Attention 层**：与普通多头注意力类似，但增加了掩码（Mask）操作，防止模型在预测第 i 个单词时获取第 i+1 个及之后的单词信息，保证预测的因果一致性。
2. **Add & Norm 层**：残差连接（$X + MaskedMultiHeadAttention(X)$）和层归一化。
3. **Cross-Attention 层**：多头交叉注意力，其 Query 来自上一个子层的输出，Key 和 Value 来自 Encoder 的编码信息矩阵 C。该层的作用是将 Decoder 生成的序列前缀与 Encoder 捕捉的输入序列上下文信息进行对齐，实现“关注输入序列中相关的部分”。
4. **Add & Norm 层**：残差连接（$X + CrossAttention(X, C, C)$）和层归一化。
5. **Feed Forward 层**：与 Encoder 中的前馈全连接层结构相同，公式为 $FFN(X) = max(0, XW_1 + b_1)W_2 + b_2$。
6. **Add & Norm 层**：残差连接（$X + FFN(X)$）和层归一化，输出当前 Decoder Block 的结果。

### （二）关键模块详解
1. **Masked Multi-Head Attention**
    - **掩码目的**：在训练阶段，Decoder 采用并行化训练，输入为完整的目标序列（如“我 有 一 只 猫”），但预测第 i 个单词时，只能利用前 i-1 个单词的信息，因此需要通过掩码掩盖第 i 个及之后的位置。
    - **掩码过程**：
        - 步骤 1：准备输入矩阵 X（目标序列的嵌入向量矩阵）和 Mask 矩阵（下三角矩阵，上三角部分为 0，下三角及对角线为 1）。例如，输入序列长度为 5 时，Mask 矩阵为：
        $\begin{bmatrix}1&0&0&0&0\\1&1&0&0&0\\1&1&1&0&0\\1&1&1&1&0\\1&1&1&1&1\end{bmatrix}$
        - 步骤 2：计算注意力分数矩阵 $QK^T$。
        - 步骤 3：将注意力分数矩阵与 Mask 矩阵按位相乘，使被掩码的位置分数变为 0（或极小值），从而在 Softmax 后权重为 0，无法获取对应位置的信息。
        - 步骤 4-5：对掩码后的分数矩阵进行 Softmax 归一化，与 Value 矩阵相乘并聚合，得到掩码注意力的输出。
2. **Cross-Attention**：该层的 Query 来自 Masked Multi-Head Attention 的输出（即已生成序列前缀的特征），Key 和 Value 来自 Encoder 的编码矩阵 C（即输入序列的上下文特征）。通过计算 Query 与 Key 的相似度，模型能够自动对齐输入序列和输出序列的相关位置，例如在机器翻译任务中，输出单词“猫”会关注输入序列中的“étudiant”（若输入为法语“Je suis étudiant”，输出为中文“我是学生”，则“学生”会关注“étudiant”）。
3. **Softmax 预测输出**：Decoder 的最后一个 Block 输出向量经 Linear 层映射到词汇表维度，再通过 Softmax 函数转换为概率分布，概率最高的单词即为当前位置的预测结果。例如，对于输出序列的第 1 个位置，Softmax 输出“我”的概率最高，第 2 个位置输出“是”的概率最高，以此类推。

## 七、Depth/width/heads trade-offs（深度、宽度、头数的权衡）
在固定算力、显存或延迟预算的情况下，Transformer 的模型容量需要在三个关键维度进行分配：深度（层数 L）、宽度（向量维度 $d_{model}$、前馈层维度 $d_{ff}$）和头数（h），三者相互牵制，需根据任务需求权衡：
1. **Depth（深度：层数 L）**
    - 优势：增加层数可以延长模型的“思考链路”，使模型能够捕捉更复杂的上下文依赖关系（如长距离语义关联），表达能力更强。
    - 劣势：层数过多会导致训练难度增加（如梯度消失、过拟合），同时推理延迟升高（需逐层计算）。
2. **Width（宽度：$d_{model}$、$d_{ff}$）**
    - 优势：增加向量维度 $d_{model}$ 或前馈层维度 $d_{ff}$ 可以提升单层网络的容量，使每个位置的向量能够编码更丰富的信息，性能提升直接。
    - 劣势：参数量和计算量通常随维度的平方增长（如 $d_{model}$ 翻倍，参数量约为原来的 4 倍），训练成本急剧增加，显存占用大幅上升。
3. **Heads（头数：h）**
    - 优势：多头注意力的头数越多，模型能够同时捕捉的视角越多，可建模的依赖关系类型越丰富（如语法结构、语义关联、指代关系等）。
    - 劣势：头数过少会导致模型无法充分捕捉多样的依赖关系；头数过多则可能产生“头冗余”，部分头的输出相似，导致计算资源浪费，且可能降低模型泛化能力。

## 八、Residual pathways & Normalization（残差连接与归一化）
残差连接和层归一化是 Transformer 能够稳定训练深度网络的关键技术，二者在每个子层（Attention 层、FFN 层）前后均有应用。
1. **Residual pathways（残差连接）**
    - 核心思想：将子层的输入直接与输出相加，即 $Output = Input + SubLayer(Input)$。
    - 作用：缓解深度网络训练中的梯度消失问题。在反向传播时，梯度可以通过残差路径直接传递到浅层，避免因层数过深导致梯度衰减至零，使深层网络能够有效训练。同时，残差连接还能保留输入的原始信息，与子层提取的特征融合，提升模型的表达能力。
2. **Normalization（层归一化）**
    - 核心思想：对每个样本的每个特征维度进行归一化，使特征分布的均值为 0、方差为 1，公式为 $LayerNorm(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$。
    - 作用：稳定网络的输入分布，减少内部协变量偏移（Internal Covariate Shift），使模型对初始化更鲁棒，加速训练收敛。与批量归一化（BatchNorm）不同，层归一化不依赖批量大小，适用于序列长度不固定的 NLP 任务。

## 九、Pre-LN/LayerNorm analysis（Pre-LN 与 Post-LN 分析）
Pre-LN 和 Post-LN 是 Transformer 中残差连接和层归一化的两种排列方式，核心区别在于归一化的执行顺序：
1. **Post-LN（原始 Transformer 结构）**
    - 结构：$SubLayerOutput = SubLayer(Input)$ → $Output = LayerNorm(Input + SubLayerOutput)$，即先执行子层计算和残差连接，再进行层归一化。
    - 特点：归一化作用于残差连接之后，能够平滑残差连接的输出，但在深层网络中容易出现训练不稳定的问题（如梯度爆炸或消失）。
2. **Pre-LN（当前主流结构）**
    - 结构：$NormalizedInput = LayerNorm(Input)$ → $SubLayerOutput = SubLayer(NormalizedInput)$ → $Output = Input + SubLayerOutput$，即先对输入进行层归一化，再执行子层计算和残差连接。
    - 特点：归一化作用于子层输入之前，使子层始终在稳定的分布上进行学习，训练稳定性显著提升，支持更深的网络结构（如 100 层以上的 Transformer）。同时，Pre-LN 在多数任务中表现更优，已成为主流的 Transformer 架构变体。

### Pre-LN 成为主流的原因
- 训练稳定性更高，对学习率更不敏感，不易出现梯度爆炸/消失。
- 支持更深的网络堆叠，能够捕捉更复杂的特征。
- 在多数 NLP 任务（如机器翻译、文本生成）中，性能优于 Post-LN 或达到相当水平，权衡效果更优。

## 十、FlashAttention（快速注意力机制）
FlashAttention 是 2022 年提出的一种优化注意力计算的算法，核心目的是加速注意力计算过程并减少内存占用，解决传统注意力机制在长序列任务中（如序列长度 N=10000）显存占用过高、计算速度慢的问题。

### （一）传统 Attention 的问题
传统注意力机制的计算过程中，需要生成中间矩阵 S（$QK^T$，维度 $N×N$）和 P（Softmax(S)，维度 $N×N$），并将其存储在 HBM（高带宽内存）中。由于 S 和 P 的维度为 $N×N$，内存复杂度为 $O(N^2)$，当 N 较大时（如 N=10000），中间矩阵的大小会达到数百 GB，远超 GPU 显存容量，导致计算无法进行。同时，HBM 与 GPU 核心之间的带宽有限，数据读写耗时较长，进一步降低了计算效率。

### （二）FlashAttention 的优化原理
FlashAttention 基于“平铺（Tiling）”和“重计算（Recomputation）”两大核心技术，利用 GPU 的 SRAM（静态随机存取存储器）进行高效计算，具体优化如下：
1. **SRAM 与 HBM 的特性对比**：GPU 的 SRAM 容量较小（如 A100 GPU 的 SRAM 为 20MB），但带宽极高（19 TB/s）；而 HBM 容量较大（如 A100 的 HBM 为 40GB），但带宽较低（1.5 TB/s）。FlashAttention 充分利用 SRAM 的高带宽特性，将注意力计算转移到 SRAM 中进行。
2. **平铺（Tiling）**：将大尺寸的 Q、K、V 矩阵切分为多个适应 SRAM 容量的小批量块（如 $B_q×d$、$B_k×d$、$B_k×d$），每次仅将一个小块加载到 SRAM 中进行计算，避免直接处理 $N×N$ 的大矩阵。
3. **重计算（Recomputation）**：前向传播时不存储中间矩阵 S 和 P，仅存储计算过程中所需的统计量（如 Softmax 中的均值 m 和方差 l）；反向传播时，利用存储的统计量和原始 Q、K、V 块，在 SRAM 中重新计算 S 和 P，避免了中间矩阵的存储和 HBM 读写操作。

### （三）FlashAttention 的性能优势
- 内存复杂度：从传统 Attention 的 $O(N^2)$ 降低为 $O(Nd)$（d 为 Q/K/V 的维度），线性依赖于序列长度 N，支持更长序列的注意力计算。
- 计算速度：由于减少了 HBM 与 GPU 核心之间的数据传输，且 SRAM 带宽更高，FlashAttention 的计算速度相比传统 Attention 提升 2-4 倍；后续推出的 FlashAttention2 进一步将速度提升 2 倍。
- 适用场景：长序列 NLP 任务（如文档翻译、长文本生成）、计算机视觉中的注意力任务（如目标检测、图像分割）等。

### （四）FlashAttention 的前向计算过程
1. 初始化并切分 Q、K、V 矩阵为适应 SRAM 容量的小块。
2. 外层循环：遍历 K、V 矩阵的所有小块。
3. 内层循环：遍历 Q 矩阵的所有小块。
4. 将当前 Q、K、V 小块加载到 SRAM 中。
5. 在 SRAM 中计算当前小块的注意力分数矩阵 $S_j = Q_jK_j^T$。
6. 对 $S_j$ 应用掩码操作（若为 Masked Attention）。
7. 计算 $S_j$ 的 Softmax 统计量 m_j（最大值）和 l_j（归一化因子）。
8. 对 $S_j$ 进行 Dropout 操作（若启用）。
9. 计算当前小块的输出 $O_j = Softmax(S_j)V_j$，并将其写入 HBM。
10. 将统计量 m_j、l_j 写入 HBM，用于反向传播重计算。