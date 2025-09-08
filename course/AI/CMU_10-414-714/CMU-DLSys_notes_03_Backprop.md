本文主要整理10-414/714 lecture3 - Manual Neural Networks / Backprop的要点。

## 3.0 Neural networks in machine learning

### **内容概括**
该幻灯片聚焦**神经网络在机器学习框架中的定位**，强调其仅构成ML算法的三大核心要素之一，需与损失函数、优化方法协同工作：
1. **神经网络的作用**：定义假设类 $h_{\theta}(x)$（即模型结构），作为输入到输出的映射函数。
2. **损失函数**：沿用交叉熵损失（Cross Entropy Loss）$\ell_{ce}$，与先前线性模型一致。
3. **优化过程**：仍采用随机梯度下降（SGD），目标是最小化训练集上的平均损失：
   $$
   \underset{\theta}{\text{minimize}}\frac{1}{m}\sum_{i=1}^{m}\ell_{ce}\left(h_{\theta}\left(x^{(i)}\right),y^{(i)}\right)
   $$
4. **关键需求**：为执行SGD，需计算损失函数对所有权重参数 $\theta$ 的梯度 $\nabla_{\theta}\ell_{ce}$。

---

### **核心要点总结**
| **要素**         | **内容**                                                                 | **与传统ML的一致性**                              |
|-------------------|--------------------------------------------------------------------------|--------------------------------------------------|
| **假设类**        | 神经网络 $h_{\theta}(x)$ 定义模型结构（如全连接网络、CNN等）             | 替代线性模型等传统假设类                          |
| **损失函数**      | 交叉熵损失 $\ell_{ce}$（分类任务标准损失）                               | 与线性分类器相同                                  |
| **优化方法**      | 随机梯度下降（SGD）                                                     | 沿用经典优化方式                                  |
| **核心挑战**      | 计算梯度 $\nabla_{\theta}\ell_{ce}$（需高效求解高维非凸优化问题）         | 神经网络参数量激增，梯度计算复杂度显著高于线性模型  |

---

### **关键结论**
1. **神经网络的本质定位**：  
   - **非独立算法**：需嵌入“损失函数+优化器”的标准ML框架中才能工作。  
   - **结构升级**：仅将假设类 $h_{\theta}(x)$ 从线性模型扩展为非线性神经网络，其余组件保持不变。  

2. **工程意义**：  
   - **梯度计算是关键瓶颈**：参数量 $\theta$ 可达百万级，需依赖**反向传播（Backpropagation）** 高效计算梯度。  
   - **框架设计统一性**：PyTorch/TensorFlow等库将损失函数、SGD优化器封装为通用模块，用户只需自定义 $h_{\theta}(x)$。  

3. **与历史内容的关联**：  
   - 延续此前“三层ML框架”（假设类、损失函数、优化器）的论述逻辑，突出神经网络的**模块化替换特性**。  

## 3.1 The gradient(s) of a two-layer network

### **内容概括**
#### **图1：输出层权重 $W_2$ 的梯度推导**
- **目标函数**：$\nabla_{W_2} \ell_{ce}(\sigma(XW_1)W_2, y)$  
- **推导逻辑**：  
  1. **链式法则分解**：  
     $$ \frac{\partial \ell_{ce}}{\partial W_2} = \frac{\partial \ell_{ce}}{\partial (\sigma(XW_1)W_2)} \cdot \frac{\partial (\sigma(XW_1)W_2)}{\partial W_2} $$  
  2. **关键中间量**：  
     - $S = \text{normalize}(\exp(\sigma(XW_1)W_2))$（Softmax输出概率）  
     - $\frac{\partial \ell_{ce}}{\partial (\sigma(XW_1)W_2)} = S - I_y$（交叉熵损失梯度，$I_y$为独热编码标签）  
  3. **最终梯度**：  
     $$ \nabla_{W_2} \ell_{ce} = \sigma(XW_1)^T (S - I_y) $$  
     - **维度匹配**：$\sigma(XW_1)^T \in \mathbb{R}^{d \times m}$, $(S - I_y) \in \mathbb{R}^{m \times k}$ → 梯度维度 $d \times k$  

#### **图2：隐层权重 $W_1$ 的梯度推导**
- **目标函数**：$\nabla_{W_1} \ell_{ce}(\sigma(XW_1)W_2, y)$  
- **推导逻辑**：  
  1. **四步链式法则**：  
     $$ \frac{\partial \ell_{ce}}{\partial W_1} = \frac{\partial \ell_{ce}}{\partial (\sigma(XW_1)W_2)} \cdot \frac{\partial (\sigma(XW_1)W_2)}{\partial \sigma(XW_1)} \cdot \frac{\partial \sigma(XW_1)}{\partial (XW_1)} \cdot \frac{\partial (XW_1)}{\partial W_1} $$  
  2. **中间梯度**：  
     - $\frac{\partial \ell_{ce}}{\partial (\sigma(XW_1)W_2)} = S - I_y$  
     - $\frac{\partial (\sigma(XW_1)W_2)}{\partial \sigma(XW_1)} = W_2$  
     - $\frac{\partial \sigma(XW_1)}{\partial (XW_1)} = \sigma'(XW_1)$（激活函数导数，如ReLU导数为阶跃函数）  
     - $\frac{\partial (XW_1)}{\partial W_1} = X$  
  3. **最终梯度**：  
     $$ \nabla_{W_1} \ell_{ce} = X^T \left[ (S - I_y) W_2^T \circ \sigma'(XW_1) \right] $$  
     - **符号说明**：$\circ$ 为逐元素乘法（Hadamard积）  
     - **维度匹配**：$X^T \in \mathbb{R}^{n \times m}$, $(S-I_y)W_2^T \in \mathbb{R}^{m \times d}$, $\sigma'(XW_1) \in \mathbb{R}^{m \times d}$ → 梯度维度 $n \times d$  

---

### **核心要点总结**
| **梯度类型** | **推导核心步骤**                                                                 | **数学形式**                                                                 | **计算关键**                                  |
|--------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------|
| **$W_2$梯度** | 1. 损失对输出求导 → $S-I_y$ <br> 2. 输出对$W_2$求导 → $\sigma(XW_1)$             | $\nabla_{W_2} = \sigma(XW_1)^T (S - I_y)$                                  | 与Softmax回归一致，仅输入替换为隐层输出      |
| **$W_1$梯度** | 1. 链式法则四层分解 <br> 2. 引入激活函数导数 $\sigma'$ <br> 3. 哈达玛积整合局部梯度 | $\nabla_{W_1} = X^T \left[ (S-I_y) W_2^T \circ \sigma'(XW_1) \right]$       | 依赖 $\sigma'$ 的局部梯度传播（反向传播本质） |

---

### **关键结论**
1. **梯度结构的对称性**：  
   - $W_2$ 梯度：**隐层输出转置** × **输出层误差**  
   - $W_1$ 梯度：**输入转置** × (**输出层误差传播至隐层** ◦ **激活函数局部梯度**)  

2. **反向传播的本质**：  
   - **误差反向传播**：$(S-I_y)$ 从输出层向隐层传递（乘以 $W_2^T$）  
   - **局部梯度激活**：$\sigma'(XW_1)$ 捕捉激活函数的敏感度（如ReLU在正值区为1）  

3. **工程实现启示**：  
   - **批量计算优势**：矩阵运算 $X^T [\cdots]$ 支持GPU并行加速（$m$个样本同时求导）  
   - **激活函数选择**：需可导（如ReLU替代Sigmoid避免梯度消失），$\sigma'$ 的计算效率影响训练速度  

4. **与深层网络的关联**：  
   - 两层网络梯度是理解反向传播的基础，$L$层网络只需递归应用链式法则  
   - $W_1$ 梯度公式中的 $ (S-I_y) W_2^T $ 对应深层网络中“梯度从上一层回传”的通用形式  


## 3.1 矩阵求导公式汇总

### **一、基础定义**
1. **布局约定**（Layout Convention）：  
   - **分母布局**（Denominator Layout）：梯度矩阵维度与 **分母变量维度相同**（主流约定）。  
   - 标量 $f$ 对矩阵 $X \in \mathbb{R}^{m \times n}$ 的梯度：  
     $$
     \nabla_X f = \frac{\partial f}{\partial X} \in \mathbb{R}^{m \times n}
     $$

---

### **二、核心公式表**
| **函数形式**                  | **梯度公式**                          | **维度验证**               | **推导逻辑**                  |
|-------------------------------|---------------------------------------|---------------------------|------------------------------|
| **线性函数**<br>$f = a^T X b$ | $\nabla_X f = a b^T$                  | $X \in \mathbb{R}^{m \times n}$<br>$\nabla_X f \in \mathbb{R}^{m \times n}$ | 标量对矩阵求导，$a,b$ 为常数向量 |
| **二次型**<br>$f = X^T A X$   | $\nabla_X f = (A + A^T) X$            | $A \in \mathbb{R}^{n \times n}$<br>$\nabla_X f \in \mathbb{R}^{n \times n}$ | 若 $A$ 对称，则 $\nabla_X f = 2AX$ |
| **矩阵乘法**<br>$Z = X W$<br>（$X$ 变量） | $\nabla_X \ell = \nabla_Z \ell \cdot W^T$ | $X \in \mathbb{R}^{m \times d}, W \in \mathbb{R}^{d \times k}$<br>$\nabla_X \ell \in \mathbb{R}^{m \times d}$ | 链式法则：$\frac{\partial \ell}{\partial X} = \frac{\partial \ell}{\partial Z} \frac{\partial Z}{\partial X}$ |
| **矩阵乘法**<br>$Z = X W$<br>（$W$ 变量） | $\nabla_W \ell = X^T \cdot \nabla_Z \ell$ | $\nabla_W \ell \in \mathbb{R}^{d \times k}$ | $\frac{\partial \ell}{\partial W} = \frac{\partial Z}{\partial W}^T \frac{\partial \ell}{\partial Z}$ |
| **逐元素函数**<br>$Z = \sigma(X)$ | $\nabla_X \ell = \nabla_Z \ell \circ \sigma'(X)$ | $\nabla_X \ell \in \mathbb{R}^{m \times n}$ | $\circ$ 为 Hadamard 积（逐元素乘） |

---

### **三、神经网络梯度特例**
#### **1. 两层网络输出层权重 $W_2$**
$$
\nabla_{W_2} \ell_{ce} = \underbrace{\sigma(XW_1)^T}_{d \times m} \underbrace{(S - I_y)}_{m \times k}
$$
- **推导**：$\ell_{ce}$ 对 $W_2$ 的梯度 = 隐层输出转置 × 输出层误差  
- **维度**：$(d \times m) \times (m \times k) \to d \times k$（匹配 $W_2$）

#### **2. 两层网络隐层权重 $W_1$**
$$
\nabla_{W_1} \ell_{ce} = \underbrace{X^T}_{n \times m} \left[ \underbrace{(S - I_y)}_{m \times k} \underbrace{W_2^T}_{k \times d} \circ \underbrace{\sigma'(XW_1)}_{m \times d} \right]
$$
- **推导**：链式法则四步分解（误差反向传播 × 激活导数）  
- **维度**：$(n \times m) \times (m \times d) \to n \times d$（匹配 $W_1$）

---

### **四、通用链式法则（矩阵版）**
若 $Y = g(X)$, $Z = h(Y)$, $\ell = f(Z)$，则：
$$
\nabla_X \ell = \left( \frac{\partial Y}{\partial X} \right)^T \nabla_Y \ell \quad \text{或} \quad \nabla_X \ell = \nabla_Y \ell \cdot \frac{\partial Z}{\partial Y} \cdot \frac{\partial Y}{\partial X}
$$
- **关键**：根据维度选择乘法顺序（左乘/右乘/转置）。

---

### **五、常见问题解答**
#### **Q1：为什么梯度公式总有转置 $T$？**  
- **答**：维度匹配的数学必然。例如 $\nabla_W \ell$ 中，若 $Z = XW$，则 $\frac{\partial Z}{\partial W} = X^T$（分母布局）。

#### **Q2：Hadamard积 $\circ$ 何时出现？**  
- **答**：当函数为**逐元素操作**时（如 $\sigma(X)$、$X \odot Y$），梯度需逐元素乘激活导数 $\sigma'(X)$。

#### **Q3：如何验证梯度维度？**  
- **法则**：$\nabla_X \ell$ 的维度必须与 $X$ 完全相同。  
  **示例**：  
  - $X \in \mathbb{R}^{3\times2}$, $\nabla_X \ell \in \mathbb{R}^{3\times2}$  
  - $W \in \mathbb{R}^{5\times1}$, $\nabla_W \ell \in \mathbb{R}^{5\times1}$

---

### **六、总结表：维度匹配模板**
| **变量类型**       | **梯度维度**          | **示例场景**               |
|--------------------|-----------------------|---------------------------|
| 标量 $c$           | $\nabla_c \ell \in \mathbb{R}$ | 偏置项 $b$ 的梯度         |
| 向量 $v \in \mathbb{R}^d$ | $\nabla_v \ell \in \mathbb{R}^d$ | 全连接层偏置向量          |
| 矩阵 $M \in \mathbb{R}^{m \times n}$ | $\nabla_M \ell \in \mathbb{R}^{m \times n}$ | 权重矩阵 $W$ 的梯度       |

> **重要提示**：实际编程中（如PyTorch），框架自动处理求导维度，但理解数学原理对调试模型至关重要。建议结合具体反向传播代码（如`loss.backward()`）加深理解。

## 3.2 Backpropagation “in general”

### **内容概括**
#### **图1：反向传播通用框架（理论核心）**
- **前向传播**：$L$层全连接网络  
  $$Z_{i+1} = \sigma_i(Z_i W_i), \quad i=1,\dots,L \quad (Z_1 = X)$$
- **梯度目标**：计算 $\frac{\partial \ell}{\partial W_i}$（损失对第 $i$ 层权重的梯度）
- **链式法则分解**：  
  $$\frac{\partial \ell}{\partial W_i} = \underbrace{\frac{\partial \ell}{\partial Z_{L+1}}}_{G_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdots \frac{\partial Z_{i+1}}{\partial W_i}$$
- **关键定义**：引入误差项 $G_{i+1} = \frac{\partial \ell}{\partial Z_{i+1}}$
- **反向迭代公式**：  
  $$G_i = G_{i+1} \cdot \sigma_i'(Z_i W_i) \cdot W_i$$

#### **图2：梯度计算的工程实现（矩阵运算）**
- **误差项维度**：$G_i = \nabla_{Z_i} \ell \in \mathbb{R}^{m \times n_i}$（$m$为批量大小，$n_i$为第$i$层维度）
- **工程化迭代公式**：  
  $$G_i = \left( G_{i+1} \circ \sigma'(Z_i W_i) \right) W_i^T \quad (\circ \text{为逐元素乘})$$
- **权重梯度计算**：  
  $$\nabla_{W_i} \ell = Z_i^T \left( G_{i+1} \circ \sigma'(Z_i W_i) \right) \in \mathbb{R}^{n_i \times n_{i+1}}$$

---

### **核心要点总结**
| **主题**                | **关键内容**                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **反向传播本质**        | 通过链式法则将损失梯度从输出层（$G_{L+1}$）逐层反向传播至各权重              |
| **误差项 $G_i$**        | - 定义：损失对第 $i$ 层输入 $Z_i$ 的梯度<br>- 物理意义：第 $i$ 层的“误差信号” |
| **反向迭代核心**        | $G_i = G_{i+1} \cdot \sigma_i'(Z_i W_i) \cdot W_i$ → 理论形式<br>$G_i = (G_{i+1} \circ \sigma') W_i^T$ → 工程实现 |
| **权重梯度公式**        | $\nabla_{W_i} \ell = Z_i^T (G_{i+1} \circ \sigma')$（$Z_i$ 为第 $i$ 层输入） |
| **激活函数导数作用**    | $\sigma'$ 控制梯度传播强度（如ReLU的导数为0或1，决定梯度通断）               |

---

### **关键结论**
#### **1. 理论与工程的衔接**
- **理论公式**：$G_i = G_{i+1} \cdot \sigma' \cdot W_i$  
  **工程实现**：$G_i = (G_{i+1} \circ \sigma') W_i^T$  
  - **差异原因**：数学推导使用标量链式法则，工程需处理**批量数据矩阵运算**  
  - **Hadamard积 $\circ$**：激活函数导数 $\sigma'$ 需与 $G_{i+1}$ 逐元素相乘（因 $\sigma$ 逐元素作用）  

#### **2. 维度验证（确保公式正确性）**
| **变量**       | **维度**          | **梯度公式维度验证**                                  |
|----------------|-------------------|-----------------------------------------------------|
| $G_{i+1}$      | $\mathbb{R}^{m \times n_{i+1}}$ | 输入误差信号                                        |
| $\sigma'(Z_i W_i)$ | $\mathbb{R}^{m \times n_{i+1}}$ | 与 $G_{i+1}$ 同维度 → 可逐元素乘                     |
| $W_i^T$        | $\mathbb{R}^{n_{i+1} \times n_i}$ | $(G_{i+1} \circ \sigma') \in \mathbb{R}^{m \times n_{i+1}}$ → 乘 $W_i^T$ 得 $G_i \in \mathbb{R}^{m \times n_i}$ |
| $\nabla_{W_i} \ell$ | $\mathbb{R}^{n_i \times n_{i+1}}$ | $Z_i^T \in \mathbb{R}^{n_i \times m}$ × $(G_{i+1} \circ \sigma') \in \mathbb{R}^{m \times n_{i+1}}$ → $n_i \times n_{i+1}$ |

## 3.3 Backpropagation: Forward and backward passes

### **内容概括**
#### **图1：反向传播算法框架（工程实现）**
- **前向传播流程**：  
  $Z_1 = X$ → $Z_{i+1} = \sigma_i(Z_i W_i)$（$i=1$到$L$）  
  - **目标**：计算各层输出 $Z_i$ 并缓存（用于反向传播）
- **反向传播流程**：  
  1. 初始化输出层梯度：$G_{L+1} = S - I_y$（Softmax输出误差）  
  2. 迭代计算：$G_i = (G_{i+1} \circ \sigma_i'(Z_i W_i)) W_i^T$（$i=L$到$1$）  
  3. 权重梯度计算：$\nabla_{W_i} \ell = Z_i^T (G_{i+1} \circ \sigma_i'(Z_i W_i))$  
- **核心思想**：链式法则 + 中间结果缓存（$Z_i$ 和 $\sigma_i'$）

#### **图2：反向传播的数学本质（理论解析）**
- **关键问题**：反向迭代 $G_i$ 的数学含义？  
- **答案**：  
  $G_i$ 的传递本质是 **向量-雅可比积（Vector-Jacobian Product, VJP）**：  
  $$G_i = G_{i+1} \cdot \underbrace{\frac{\partial Z_{i+1}}{\partial Z_i}}_{\text{雅可比矩阵}}$$  
  - 雅可比矩阵 $\frac{\partial Z_{i+1}}{\partial Z_i}$ 的维度为 $(n_{i+1} \times n_i)$  
  - $G_{i+1}$ 作为行向量左乘该矩阵，实现误差反向传播  
- **泛化意义**：  
  该过程可推广至任意计算图，构成 **自动微分（Automatic Differentiation）** 的基础。

---

### **核心要点总结**
| **主题**                | **关键内容**                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **前向传播**            | 逐层计算输出 $Z_i$ 并缓存，为反向传播提供中间结果                              |
| **反向传播**            | 1. 初始化 $G_{L+1} = S - I_y$（输出层误差）<br>2. 迭代计算 $G_i$（损失对 $Z_i$ 的梯度）<br>3. 计算 $\nabla_{W_i} \ell$（损失对权重的梯度） |
| **$G_i$ 的物理意义**     | 第 $i$ 层的 **反向传播误差信号**，指导权重更新方向                            |
| **向量-雅可比积（VJP）** | $G_i = G_{i+1} \cdot \frac{\partial Z_{i+1}}{\partial Z_i}$（核心数学操作） |
| **自动微分关联**        | 反向传播是自动微分的特例，通过计算图的局部导数链式传播梯度                      |

---

### **关键结论**
#### **1. 算法设计精髓**
- **缓存中间结果**：前向传播存储 $Z_i$ 和 $W_i$，避免反向传播重复计算（时间复杂度 $O(L)$）。  
- **模块化计算**：每层仅需实现两个函数：  
  - **前向函数**：$Z_{i+1} = \text{forward}(Z_i, W_i)$  
  - **反向函数**：$(G_i, \nabla_{W_i} \ell) = \text{backward}(G_{i+1}, Z_i, W_i)$  

#### **2. 向量-雅可比积的工程实现**
以全连接层为例：  
- **雅可比矩阵**：$\frac{\partial Z_{i+1}}{\partial Z_i} = \sigma_i'(Z_i W_i) \cdot W_i^T$  
- **VJP计算**：  
  ```python
  def backward(G_next, Z, W):
      d_act = sigma_prime(Z @ W)        # 计算激活函数导数 (m × n_{i+1})
      dW = Z.T @ (G_next * d_act)       # 权重梯度：Z^T (G_{i+1} ◦ σ')
      G_prev = (G_next * d_act) @ W.T   # 反向传播误差：G_i = (G_{i+1} ◦ σ') W^T
      return G_prev, dW
  ```

#### **3. 与自动微分的关联**
- **计算图视角**：神经网络是计算图，节点为运算（矩阵乘、激活函数），边为数据流。  
- **反向模式微分**：  
  从输出端开始，逐节点计算 **输出梯度**（$G_i$）并传播，与反向传播完全一致。  
