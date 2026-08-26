# Training a Deep Neural Network: Forward and Backward Pass

## 1. Setup and Notation

Consider a feedforward network with $L$ layers. For layer $l = 1, \dots, L$:

- $\mathbf{W}^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$ — weight matrix
- $\mathbf{b}^{[l]} \in \mathbb{R}^{n_l}$ — bias vector
- $\mathbf{a}^{[0]} = \mathbf{x}$ — input
- $\sigma^{[l]}(\cdot)$ — activation function of layer $l$

Let $\theta = \{\mathbf{W}^{[l]}, \mathbf{b}^{[l]}\}_{l=1}^{L}$ denote all trainable parameters.

---

## 2. Forward Propagation

For each layer $l = 1, \dots, L$, compute the pre-activation and activation:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}\left(\mathbf{z}^{[l]}\right)
$$

The final output is $\hat{\mathbf{y}} = \mathbf{a}^{[L]}$.

**Loss function.** Given target $\mathbf{y}$, define a scalar loss:

$$
\mathcal{L} = \ell(\hat{\mathbf{y}}, \mathbf{y})
$$

e.g., mean squared error $\ell = \frac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$, or cross-entropy $\ell = -\sum_k y_k \log \hat{y}_k$.

For a batch of $m$ examples, the empirical risk is:

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \ell\left(\hat{\mathbf{y}}^{(i)}, \mathbf{y}^{(i)}\right)
$$

---

## 3. Backward Propagation (Backpropagation)

The goal is to compute $\frac{\partial J}{\partial \mathbf{W}^{[l]}}$ and $\frac{\partial J}{\partial \mathbf{b}^{[l]}}$ for every layer, using the **chain rule**, propagating error signals from the output layer backward.

**Output layer error:**

$$
\boldsymbol{\delta}^{[L]} = \nabla_{\mathbf{a}^{[L]}} \mathcal{L} \odot \sigma^{[L]\prime}\left(\mathbf{z}^{[L]}\right)
$$

where $\odot$ denotes the Hadamard (elementwise) product.

**Recursive error propagation** (for $l = L-1, \dots, 1$):

$$
\boldsymbol{\delta}^{[l]} = \left(\left(\mathbf{W}^{[l+1]}\right)^{\top} \boldsymbol{\delta}^{[l+1]}\right) \odot \sigma^{[l]\prime}\left(\mathbf{z}^{[l]}\right)
$$

**Parameter gradients:**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} \left(\mathbf{a}^{[l-1]}\right)^{\top}, \qquad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}
$$

Derivation sketch (chain rule at layer $l$):

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} \cdot \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{W}^{[l]}}
= \boldsymbol{\delta}^{[l]} \cdot \left(\mathbf{a}^{[l-1]}\right)^{\top}
$$

since $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$.

For a minibatch, average over examples:

$$
\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \boldsymbol{\delta}_i^{[l]} \left(\mathbf{a}_i^{[l-1]}\right)^{\top}
$$

where the subscript $i$ indexes the $i$-th training example in the minibatch (i.e., $\boldsymbol{\delta}_i^{[l]}$ and $\mathbf{a}_i^{[l-1]}$ are the error signal and activation computed for example $i$).

---

## 4. Parameter Update (Gradient Descent)

With learning rate $\eta$, the vanilla update rule is:

$$
\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial J}{\partial \mathbf{W}^{[l]}}, \qquad
\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial J}{\partial \mathbf{b}^{[l]}}
$$

More generally, for an optimizer update function $U$ (SGD with momentum, Adam, etc.):

$$
\theta_{t+1} = \theta_t - \eta \, U\left(\nabla_\theta J(\theta_t), t\right)
$$

For example, **Adam**:

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla_\theta J, \qquad
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla_\theta J)^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \qquad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
$$

---

## 5. Full Algorithm (One Training Step)

1. **Forward pass:** compute $\mathbf{z}^{[l]}, \mathbf{a}^{[l]}$ for $l=1,\dots,L$; compute $J(\theta)$.
2. **Backward pass:** compute $\boldsymbol{\delta}^{[L]}$, then $\boldsymbol{\delta}^{[l]}$ recursively for $l = L-1,\dots,1$; compute $\partial J/\partial \mathbf{W}^{[l]}$, $\partial J/\partial \mathbf{b}^{[l]}$.
3. **Update:** apply the optimizer rule to update $\theta$.
4. Repeat over minibatches/epochs until convergence (e.g., $\|\nabla_\theta J\|$ small or validation loss stops improving).

This is the core computational graph underlying **automatic differentiation** as implemented in frameworks like PyTorch/TensorFlow — the chain rule above is applied generically to any differentiable computation graph, not just this fully-connected layer structure (the same principle extends to convolutions, attention, normalization layers, etc., with appropriately defined local Jacobians).

---

## 6. Shapes of the Activations $\mathbf{a}$

### Single example case

For layer $l$, the input to that layer is the activation from the previous layer:

$$
\mathbf{a}^{[l-1]} \in \mathbb{R}^{n_{l-1} \times 1}
$$

where $n_{l-1}$ is the number of units (neurons) in layer $l-1$.

- At the very start, $\mathbf{a}^{[0]} = \mathbf{x} \in \mathbb{R}^{n_0 \times 1}$, where $n_0$ is the input feature dimension (e.g., 784 for flattened MNIST images).
- This makes the matrix multiplication $\mathbf{W}^{[l]} \mathbf{a}^{[l-1]}$ consistent:

$$
\underbrace{\mathbf{W}^{[l]}}_{n_l \times n_{l-1}} \cdot \underbrace{\mathbf{a}^{[l-1]}}_{n_{l-1} \times 1} = \underbrace{\mathbf{z}^{[l]}}_{n_l \times 1}
$$

### Minibatch case

In practice you process $m$ examples at once, stacked as columns:

$$
\mathbf{A}^{[l-1]} \in \mathbb{R}^{n_{l-1} \times m}
$$

Then the forward equation becomes:

$$
\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]} \in \mathbb{R}^{n_l \times m}
$$

Here $\mathbf{b}^{[l]} \in \mathbb{R}^{n_l \times 1}$ is broadcast (added to every column).

### Summary table

| Object | Shape (single example) | Shape (batch of $m$) |
|---|---|---|
| $\mathbf{a}^{[l-1]}$ | $n_{l-1} \times 1$ | $n_{l-1} \times m$ |
| $\mathbf{W}^{[l]}$ | $n_l \times n_{l-1}$ | (unchanged) |
| $\mathbf{z}^{[l]}$, $\mathbf{a}^{[l]}$ | $n_l \times 1$ | $n_l \times m$ |

**Note on convention:** Many frameworks (PyTorch, TensorFlow) instead use the transposed convention $\mathbf{A}^{[l-1]} \in \mathbb{R}^{m \times n_{l-1}}$ (rows = examples, columns = features), in which case the forward pass is written as $\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \left(\mathbf{W}^{[l]}\right)^{\top} + \mathbf{b}^{[l]}$. Both are mathematically equivalent — just a transpose of the same computation — but it's worth knowing which convention a given codebase uses before implementing backprop by hand.
