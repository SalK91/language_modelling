## Tensors and PyTorch

Tensors are the basic building block for storing everything in deep learning: parameters, gradients, optimizer state, data, and activations. Almost all of these are stored as floating-point numbers.

By default, tensors live in CPU memory. To leverage the massive parallelism of GPUs, we move them to GPU memory via `.to(device)` or `.cuda()`.

### What is a Tensor?

PyTorch tensors are pointers into allocated memory, plus metadata describing:

- `shape`
- `stride` — how many elements to skip to move along each axis
- `dtype`
- `device`

Most tensors are created from performing operations on other tensors. Each operation has some memory and compute consequence. Many operations simply provide a different *view* of the tensor. This does not make a copy, and therefore mutations in one tensor affect the other.

### Optional math example

A tensor $x \in \mathbb{R}^{m \times n}$ is stored as a contiguous 1-D block of memory, with metadata describing how to interpret it. The stride vector $s$ defines indexing via:

$
x[i, j] = \text{memory}[i \cdot s_0 + j \cdot s_1]
$

If a view operation reshapes $x$ without copying, the underlying memory remains the same:

$
x' = \text{view}(x), \quad \text{ptr}(x') = \text{ptr}(x)
$

## FLOPs and Performance in Deep Learning

A FLOP (floating-point operation) is a basic arithmetic operation (e.g., addition or multiplication) on floating-point numbers. It is a standard measure of computational work in deep learning.

#### Example: Linear Layer FLOPs

Suppose we apply a linear model to a batch of $B$ vectors of dimension $D$, mapping to $K$ outputs:

```python
B = 16384  # Batch size (e.g., number of tokens)
D = 32768  # Input dimension
K = 8192   # Output dimension

device = get_device()
x = torch.ones(B, D, device=device)
w = torch.randn(D, K, device=device)
y = x @ w
```
 
Each output $y_{ik} = \sum_j x_{ij} w_{jk}$ requires $D$ multiplications
and $D - 1$ additions.

Approximate total FLOPs:

$$
FLOPs = 2  \times B  \times D  \times K
$$

### Other FLOPs Estimates

-   Elementwise operation (e.g., ReLU on $m imes n$ matrix): $mn$ FLOPs
-   Matrix addition ($m \times n + m \times n$): $mn$ FLOPs
-   Normalization layers: $O(mn)$ FLOPs

In general, matrix multiplications dominate FLOPs in large models like
Transformers.

### Interpreting FLOPs in LLMs

-   $B$ is number of tokens or batch size
-   $D \times K$ is number of parameters in a linear layer
-   Forward pass FLOPs: $2 \cdot B \cdot D \cdot K$
-   Total FLOPs: Sum over all layers, multiplied by batch size and sequence length

### Rule of thumb

$$FLOPs \approx 2 \cdot    \text{#tokens} \cdot  \text{#parameters (per layer)}$$

### Model FLOPs Utilization (MFU)

Definition:

$$
MFU = \frac{\text{Actual FLOPs/sec}}{\text{Peak Theoretical FLOPs/sec}}
$$

MFU indicates how closely the training process approaches the maximum
compute capability of the hardware.

#### Why MFU is not always 1

-   Non-matmul operations (activations, normalization, data movement)
    are often memory-bound
-   Kernel launch overhead from many small kernels
-   Memory bandwidth bottlenecks
-   Irregular workloads (uneven tensor sizes, short sequences)
-   Communication overhead in multi-GPU setups (sync, data transfer)

#### What is a good MFU?

Typically:

$$
MFU \ge 0.5   \text{ is considered good}
$$

MFU improves when:

-   Matrix multiplications dominate the workload
-   Batch sizes and sequence lengths are large
-   Code is optimized for the hardware

## Mixed-Precision Training

### Floating-Point Formats in GPUs

#### Floating-point 101
A binary floating-point number is encoded as:

$$\underbrace{\text{sign}}_{\;1\;\text{bit}}\;
\underbrace{\text{exponent}}_{\;e\;\text{bits}}\;
\underbrace{\text{mantissa / significand}}_{\;m\;\text{bits}}
$$


| Format | Bits | Exponent | Mantissa | Approx. Range |
|---|---:|---:|---:|---|
| FP32 (single) | 32 | 8 | 23 | $10^{-45}$ – $10^{38}$ |
| FP16 (half) | 16 | 5 | 10 | $2 \times 10^{-14}$ – $2 \times 10^{15}$ |

Precision vs. range: Reducing the mantissa increases spacing between representable values, so small increments (e.g. $1.0001$) round to $1.0$ in FP16.  
A narrower exponent field also shrinks dynamic range, increasing overflow/underflow risk.

### FP32 vs. FP16 in Neural-Network Training

- Default FP32 training: Parameters and gradients are stored in FP32. High memory usage $\rightarrow$ risk of OOM on large models.

- Naïve FP16 swap: Reduces memory and bandwidth by 50%, but:
    - Gradients may underflow to zero
    - Weight updates lose precision $\rightarrow$ poor convergence

### Mixed-Precision Training: Core Idea
Keep critical operations in FP32 while using FP16 where safe:

1. Maintain a set of FP32 master weights
2. Cast a working copy of the model to FP16; do the forward pass there.
3. Back-propagate (initial gradients in FP16)
4. Convert (copy) gradients to FP32
5. Update the FP32 master weights with your optimizer (SGD/Adam, etc.).
6. Cast the updated weights back to FP16 for the next forward pass.

### Preventing Gradient Underflow: Dynamic Loss Scaling

Even with mixed precision, tiny gradients can vanish. A practical recipe, adapted from:

Procedure:

- Choose a loss‑scaling factor $S$ (e.g.\ powers of two).\footnote{Modern frameworks adjust $S$ automatically (“dynamic” scaling).}
- Multiply the computed loss by $S$ before back‑propagation.
- Compute FP16 gradients of the scaled loss.
- Convert gradients to FP32 and divide them by $S$.
- Proceed with the usual FP32 weight update \& casting loop above.


![Pytorch - Mixed Precision Training](images/mixedprecision.png)

### Benefits

- $\approx 2\times$ faster math on Tensor Cores (Volta/Ampere+)
- $\approx 2\times$ lower memory footprint, enabling larger batch sizes or models.
- Same convergence as FP32 when loss scaling is correct

### Caveats

-  Verify numerical stability on your model; some niche layers (e.g. custom CUDA kernels) may not be FP16‑safe.
- Loss scaling adds minor overhead if done manually; use the built‑in API of your deep‑learning framework whenever possible.


## BFloat16

Google Brain developed bfloat (brain floating point) in 2018 to address this issue. bfloat16 uses the same memory as float16 but has the same dynamic range as float32! The only catch is that the resolution is worse, but this matters less for deep learning.

Core idea: BFloat16 (16-bit float) keeps the same exponent size as FP32, so it has a very wide range but with lower precision (7 mantissa bits).  

This makes it:

- Fast and memory-efficient like FP16
- Avoids gradient underflow
- No loss scaling required

#### Implementation steps

1. Keep a set og FP32 master weights
2. Cast model to BFloat16 for forward/backward passes(`model.to(torch.bfloat16)`)
3. Gradients accumulate in FP32 by default
4. Update FP32 master weights as usual
5. Cast updated weights back to BFloat16 for next step


No loss scaling required, BFloat16 simplifies mixed-precision training by providing FP32-like range with FP16-like speed and memory savings.


## Multi-GPU Training: From DDP to ZeRO

1. Single-GPU Setup: In a typical mixed-precision setup:

    - Model parameters: Stored in FP16 on GPU VRAM.
    - Optimizer state (FP32):
        - Master weights (for precision-preserving updates)
        - Momentum buffers (e.g., Adam)
        - Variance estimates (e.g., Adam)

 
2. Distributed Data Parallel (DDP) — Baseline: DDP replicates the entire model and optimizer state across all GPUs and splits the input data:
    
    - Each GPU computes a forward and backward pass on its mini-batch.

    - During backpropagation, gradients are synchronized across all GPUs using an AllReduce.

    - Each GPU independently applies the optimizer update to its local copy of the model.

Communication cost: Each gradient (typically in FP32 unless cast to FP16) is sent across GPUs. The bandwidth cost is roughly ∼4 bytes per parameter (or ∼2 bytes with
FP16 compression)

Limitation: Memory overhead scales poorly — each GPU stores a full copy of model parameters and optimizer state.

## ZeRO: Zero Redundancy Optimizer (DeepSpeed)

ZeRO removes the major inefficiency of Distributed Data Parallel (DDP), where every GPU keeps a full copy of model parameters, gradients, and optimizer states. Instead, ZeRO partitions (shards) training states across GPUs so that each device holds only the slice of memory it actually needs, while still participating in full model training.


### ZeRO Stage-1: Optimizer State Sharding

- Each GPU stores:
    - The full FP16 model
    - Only a shard of the optimizer state
- Each GPU computes gradients on its data shard
- Gradients are reduce-scattered so each GPU receives only the gradients for its parameter shard
- Each GPU updates its own optimizer shard and corresponding parameters
- Updated parameters are all-gathered so all GPUs hold a synchronized model

### ZeRO Stage-2: Optimizer State + Gradient Sharding

Builds upon Stage-1 by also sharding gradients:

- Gradients are never fully instantiated in memory
- During backward pass:
    - Each GPU computes gradients for a layer
    - Immediately reduces gradients to the GPU responsible for that parameter shard
    - Frees local gradient memory once sent
- After backpropagation, each GPU:
    - Updates its optimizer and parameter shards locally
    - Parameters are synchronized via all-gather before the next forward pass

Benefit: significantly reduces memory footprint — gradients and optimizer states are distributed.



## ZeRO Stage-3: Full Model, Gradient, and Optimizer Sharding

In this final stage:

- Model parameters are sharded across GPUs — no GPU stores the full model
- Parameters are materialized just-in-time during forward/backward and deallocated afterward
- Training requires orchestration of:
    - Parameter gathering
    - Gradient sharding and reduction
    - Activation checkpointing (optional but common)
- Implemented in PyTorch via `torch.distributed.fsdp` (Fully Sharded Data Parallel)

Use case: enables training of models with hundreds of billions to trillions of parameters on commodity GPU clusters.

| Training Strategy | Model Sharded | Gradients Sharded | Optimizer Sharded |
|---|---|---|---|
| DDP (Baseline) | $\times$ | $\times$ | $\times$ |
| ZeRO Stage-1 | $\times$ | $\times$ | $\checkmark$ |
| ZeRO Stage-2 | $\times$ | $\checkmark$ | $\checkmark$ |
| ZeRO Stage-3 | $\checkmark$ | $\checkmark$ | $\checkmark$ |

## Key Takeaways

- ZeRO reduces memory consumption linearly with GPU count.
- Enables training of large models without sacrificing batch size or needing model parallelism.
- Fully Sharded Data Parallel (FSDP) in PyTorch and DeepSpeed provide user-friendly APIs to leverage ZeRO at scale.