> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/435908830)

本篇文章是深入浅出 GPU 优化系列的第两个专题，主要是**介绍如何对 GPU 中的矩阵乘法（GEMM）进行优化**。目前针对 GEMM 的优化，网络上已经有非常多的教程和示例了。大部分的重要资料我都看了看。但总的来说，还是不够接地气，然后理解起来还是会比较费解。所以希望写这么一篇文章，尽可能地去把 GPU 的 GEMM 优化说清楚，说明白。然后让小白读者也能通过这么一两篇文章去更好地了解 GEMM 优化的相关技术。

不像上次的 reduce 优化一样，能一篇文章说完。这次的 GEMM 优化会分为三个部分。**第一个部分只说优化思路和分析**，没有任何代码，这么做考虑也是为了减轻读者的负担，看代码太累，**尽可能地让读者先明白原理，为什么要这么做**。**第二个部分是对代码的详细解析，这个里面就是一行一行地去分析代码**。因为之前的很多博客进行了分析，但是代码本身并没有开源，或者说开源了代码，但没有解析，看起来太累了。我希望提供一个尽可能详细的代码解析，读者看完之后能明白相关优化技巧，并且可以直接把代码拿去验证使用。**第三个部分主要涉及到汇编器**，最重要的是说明在 NV 的卡上，怎么去解决寄存器的 bank 冲突来获取极致的性能。

本篇文章是 **GEMM 优化的第一个部分**，在这篇文章中，只说**优化思路和分析**。

前言
--

在高性能领域，对于**矩阵乘（GEMM）的优化**是一个非常重要的课题。GEMM 可以非常广泛地应用于航空航天、流体力学等科学计算领域，这也是之前 HPC 的主要应用场景。后来深度学习开展地如火如荼，由于对高算力的需要，也成为 HPC 的主要应用场景之一。这些年涌现了一系列的深度学习模型。模型里面最耗时的东西，包括卷积、全连接层、attention，都可以转换成 GEMM 操作。所以说，GEMM 优化的重要性，怎么突出都不过分。

目前网上能找到的针对 GEMM 优化的资料主要有这么几个方面：  
**一、论文**，目前针对 GPU 进行 GEMM 优化的论文非常多，这里主要推荐 [Understanding the GPU Microarchitecture](https://link.zhihu.com/?target=https%3A//dl.acm.org/doi/10.1145/3018743.3018755) 和 [Fast implementation of dgemm on fermi gpu](https://link.zhihu.com/?target=https%3A//ieeexplore.ieee.org/document/6114452) 以及 [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.06826)。这几篇论文在业界都比较有影响力，就是代码开源方面做的不算太好。**二、官方博客**，主要是 [CUTLASS](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/blog/cutlass-linear-algebra-cuda/) 和 [NervanaSystems-SGEMM 优化](https://link.zhihu.com/?target=https%3A//github.com/NervanaSystems/maxas/wiki/SGEMM)。还有前段时间旷视发的文章 [CUDA 矩阵乘法优化](https://zhuanlan.zhihu.com/p/410278370)，写的都很详细。**三、github** 的一些 demo，代码量不大，看起来比较舒服。我是看了这两个，

[demo1](https://link.zhihu.com/?target=https%3A//github.com/Cjkkkk/CUDA_gemm)[demo2](https://link.zhihu.com/?target=https%3A//github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs)

demo1 代码写的好理解一些，但是优化工作没做完全，没有做到 prefetch。demo2 是效果很好，11 个优化技巧，不断逼近 cublas。但是代码真的看起来比较难受，最重要的很多参数写死了，不好去调。

总而言之，目前列举的上述资料存在着这么两个问题：一、文档方面，读起来还是比较费劲，对于小白来说，还是不够简单不够傻，看起来太累了；二、代码方面，要么是没公开代码，要么是代码太多了，看不下去；还有的就是代码可读性很强，但是优化工作还不是特别深，或者就是代码优化做的很好，但是可读性差了。方方面面总是有点欠缺，所以希望能够写一篇尽可能地在文档上简单明了，在代码上详细且可读性好的文章。当然，这是一个逐步迭代的过程，所以这篇文章也会持续进行更新哈。

本篇文章主要是采纳了 cutlass 的行文思路，主要介绍 GEMM 中的数据分块和如何在多级存储进行数据搬运。这也是 **HPC 优化的核心思想，怎么样让数据放在更近的存储上来掩盖计算的延时，从而减少存储墙的影响**。文章分为四个方面进行叙述，首先介绍在 global memory 层面如何进行分块以及数据搬运，随后介绍在 shared memory 层面如何进行分块以及数据搬运，而后介绍在 register 层面如何进行分块以及避免 bank 冲突，最后介绍如何进行 prefetch 以更好地掩盖访存时延。

从 global memory 到 shared memory
-------------------------------

> 为什么全部的剪藏软件对数学公式的支持都这么糟糕...

> prompt：请你帮我清理数学公式的双重重复，返回给我内嵌latex的md源代码

假设有矩阵 A,B，需要计算矩阵 A 和 B 的乘，即矩阵 C。A、B、C 三个矩阵的维度分别为，， $m*k，k*n，m*n$，且三个矩阵中的数据都是单精度浮点数。对于 C 中每一个元素，C[i][j]，可以看作是 A 的一行和 B 的一列进行一次归约操作。采用最 naive 的 GEMM 算法，在 GPU 中，一共开启 $m*n$ 个线程，每个线程需要读取矩阵 A 的一行与矩阵 B 的一列，而后将计算结果写回至矩阵 C 中。因而，完成计算一共需要从 global memory 中进行 $2mnk$ 次读操作和 m*n 次写操作。大量的访存操作使得 GEMM 效率难以提高，因而考虑 global memory 中进行分块，并将矩阵块放置到 shared memory 中。其示意图如下：

![](https://pica.zhimg.com/v2-33eeaaeb4298ee311e1b6e97fb9a3c84_r.jpg)

对 global memory 进行分块的 GEMM 算法示意图见上图右侧。

- 首先将 A、B、C 三个矩阵划分为多个维度为 $bm \times bk$、$bk \times bn$、$bm \times bn$ 的小矩阵块。三个矩阵形成 $M \times K$、$K \times N$、$M \times N$ 的小矩阵网格。
    - b可以理解为类似batch之类的东西吗
    - 其中，$M = m/bm$，$N = n/bn$，$K = k/bk$。
    - 随后在 GPU 中开启 $M \times N$ 个 block，每个 block 负责 C 中一个维度为 $bm \times bn$ 的小矩阵块的计算。
        - 这里, aka 内部循环里, 继续调用正常矩阵乘
- 计算中一共有 K 次迭代，每一次迭代都需要读取 A 中一个维度为 $bm \times bk$ 的小矩阵块和 B 中一个维度为 $bk \times bn$ 的小矩阵块，并将其放置在 shared memory 中。
- 因而，完成 C 中所有元素的计算一共需要从 global memory 中读取 $M \times N \times K \times (bm \times bk + bk \times bn)$，即 $m \times n \times k \left( \frac{1}{bm} + \frac{1}{bn} \right)$ 个单精度浮点数。
    - 访存量计算，$m \times n \times k$是分块的数目，$\frac{1}{bm} + \frac{1}{bn}$是每个块儿读取的数据数

> 但是我还是没get到为什么这么计算，结果仍然正确 $\to$ 应该是分块之间累加，然后等效图左边的计算

> 或许我需要一个更直观的图解，以及习惯这个提法的存在


相比于 naive 的 GEMM 算法，访存量减少为原来的 $\frac{1}{2} \left( \frac{1}{bm} + \frac{1}{bn} \right)$。通过 global memory 中分块算法极大地减少了对 global memory 的访存量。并且，相比于 naive 算法，对 global 进行分块可以更充分地利用数据局部性。在 naive 算法中，每一个线程都需要直接从 global memory 中取数，其时延非常长，计算性能非常差。而进行分块后，将维度为 $bm \times bk$、$bk \times bn$ 的小矩阵块先存储到 shared memory 之中。而后计算单元进行计算时可以直接从 shared memory 中取数，大大减少了访存所需要的时延。


从 shared memory 到 register
--------------------------

随后，我们进一步考虑从 shared memory 到 register 的过程。在这里，只分析**一个 block** 中的计算。当进行 K 轮迭代中某一轮迭代时，GPU 将维度为， $bm*bk，bk*bn\$bm*bk，bk*bn\ 的小矩阵块存储到 shared memory 中，而后各个线程将 shared memory 中的数据存入 register 中进行计算。

> 这里和上面说的一样，每个block里还是朴素乘法...

![](https://pic4.zhimg.com/v2-09215423d4c157b13486945a37614319_r.jpg)

在 **不对 shared memory 分块** 时，一个 block 中含有 $bm \times bn$ 个线程，**每一个线程负责 C 中一个元素的计算**。则一个 block 一共需要对 shared memory 进行 $2 \times bm \times bn \times bk$ 次读操作。

而后 **考虑对 shared memory 进行分块**，对 $bm \times bn$ 的小矩阵进行再一次划分，将其划分为多个维度为 $rm \times rn$ 的子矩阵。则一个 block 需要负责 $X \times Y$ 个子矩阵，其中 $X = \frac{bm}{rm}$，$Y = \frac{bn}{rn}$。随后，在一个 block 中开启 $X \times Y$ 个线程，**每个线程负责一个维度为 $rm \times rn$ 的子矩阵的计算**。

> idea和上面完全类似

在计算中，一个 block 一共需要从 shared memory 读取 $X \times Y \times (rm + rn) \times bk$，即 $bm \times bn \times bk \left( \frac{1}{rm} + \frac{1}{rn} \right)$ 个单精度浮点数。相比于未分块的算法，对于 shared memory 中的访存量减少为原来的 $\frac{1}{2} \left( \frac{1}{rm} + \frac{1}{rn} \right)$。并且，由于将数据放入 register 中，可以直接对数据进行运算，减少了从 shared memory 中取数的时延。

register 分块
-----------

在这里，我们考虑最后一层，即寄存器中的计算，并且只分析一个线程。在完成以上的过程后，对于一个线程而言，它现在拥有 $rm$ 个 A 矩阵的寄存器值，$rn$ 个 B 矩阵的寄存器值，以及 $rm \times rn$ 个 C 矩阵的寄存器值。通过这些寄存器的值，需要计算 $rm \times rn$ 个数。这需要 $rm \times rn$ 条 FFMA 指令。

### sidenote:FFMA

FFMA 指令是“Fused Multiply-Add”的缩写，通常用于浮点运算中的一种指令。它的主要功能是执行以下操作：

```
result = (a * b) + c
```

与传统的乘法和加法操作相比，FFMA 指令可以在单个步骤中完成乘法和加法，从而提高运算效率和精度，减少舍入误差。

#### FFMA 指令的特点

1. **高效性**：通过将乘法和加法结合在一起，可以减少计算的时间和资源消耗。
2. **精度**：减少了中间结果的舍入，从而提高了计算的精度。
3. **硬件支持**：许多现代处理器和图形处理单元（GPU）都支持这一指令，特别是在高性能计算和图形渲染中。

#### 应用场景

- **科学计算**：在需要高精度浮点运算的领域，如物理模拟和计算流体动力学。
- **图形处理**：在渲染图形和视频处理时，通过减少计算步骤提高性能。

FFMA 是一种非常有用的指令，尤其是在对性能和精度有高要求的应用中。

回到正文分割线

这时候会涉及到寄存器的 bank conflict。在 NV 的 GPU 中，每个 SM 不仅会产生 shared memory 之间的 bank 冲突，也会产生寄存器之间的 bank 冲突。这一点对于计算密集型的算子十分重要。像 shared memory 一样，寄存器的 Register File 也会被分为几个 bank，如果一条指令的源寄存器有 2 个以上来自同一 bank，就会产生冲突，指令会重发射，浪费一个 cycle。

假设对于这个线程来说，$rm=4$，$rn=4$，并且计算 C 的寄存器以非常 naive 的方式分配。则需要产生 16 条 FFMA 指令，列举如下：

```
FFMA R0, R16, R20, R0
FFMA R1, R16, R21, R1
……

```

寄存器 bank 冲突：
在 GPU 的每个流处理器（SM）中，寄存器文件被划分为多个 bank。这意味着每个 bank 可以同时处理多个寄存器的读写。
如果一条指令的源寄存器有多个来自同一 bank 的寄存器，就会产生冲突，这会导致指令需要重发射，从而浪费一个周期（cycle）。 

> 就，多个sm访问同一块共享内存，导致访问串行化 bank conflict

![](https://pica.zhimg.com/v2-3d88904ca149dd4cd563036771f87796_r.jpg)

可以从中看出，这会产生大量的 register bank 冲突，所以需要对参与计算的寄存器重新进行分配和排布, 如上图右侧所示。在有些地方，这种方式也可以叫做 register 分块。

> 这图一点也看不懂... orzzz

数据的 prefetch
------------

最后，我们来讲讲如何通过对数据进行 prefetch 来减少访存的 latency。回顾 GEMM 的过程，我们可以仔细看看访存的 latency 是如何导致性能下降的。

**对于一个 block 而言**，需要计算一个 $bm \times bn$ 的矩阵块，这时需要进行 K 次迭代，每次迭代都需要先将来自 A 和 B 的两个小块送到 shared memory 中再进行计算。然而，从 global memory 中访存实际上是非常慢的，这导致了 latency。虽然 GPU 中可以通过 block 的切换来掩盖这种 latency，但是由于分配的 shared memory 比较多，活跃的 block 并不是很多，这种延时很难被掩盖。

**对于一个 thread**，需要计算一个 $rm \times rn$ 的小矩阵，但必须先将数据从 shared memory 传到寄存器上，才能开始进行计算。因此，在每次迭代中，计算单元需要停下来等待，无法充分利用计算资源。

为此，需要进行数据的 prefetch 来尽可能地掩盖这种 latency。其思想比较简单：需要多开一个 buffer，进行**读写分离**。示意图如下。当 block 进行第 2 轮迭代时，需要对 $A_2$ 和 $B_2$ 进行计算。在计算单元进行计算的同时，我们将 $A_3$ 和 $B_3$ 提前放置到 shared memory。随后，在进行第 3 轮迭代时，就可以直接对 shared memory 中的 $A_3$ 和 $B_3$ 进行计算，而不需要等待从 global memory 搬运到 shared memory 的时间。寄存器上的 prefetch 也是同理。

![](https://pic1.zhimg.com/v2-bf58cbda60ee4eed6fcd03ca6a3fe35e_b.jpg)

总结
--

GEMM 的优化思想，基本上就是这么几方面的内容。希望大家通过介绍能够对 GEMM 的优化有一个比较直观且具体的理解。（感觉写的还是有点乱，图也没画的太好，大家谅解）。当然，看完这些，要开始写代码的时候，大家还是会比较懵，也不知道这些东西应该怎么实现。现在写了详细的解析，也就是 GEMM 优化（二）。去实现了上述优化技巧和细致地分析每一行代码，大家可以看一看。

[有了琦琦的棍子：深入浅出 GPU 优化系列：GEMM 优化（二）](https://zhuanlan.zhihu.com/p/442930482)

第三部分中，更细粒度的 CUDA C 代码调优和关于汇编代码的调优，也已经给出。

[有了琦琦的棍子：深入浅出 GPU 优化系列：GEMM 优化（三）](https://zhuanlan.zhihu.com/p/481600052)

最后，**感谢大家看到这里，有什么问题欢迎跟我讨论哈**。关于 GPU 的优化，打算写一个系列，说说 GPU 优化的一些经典问题和优化技巧。不过最近工作也比较忙，更新估计很慢。之前已经写完了

[有了琦琦的棍子：深入浅出 GPU 优化系列：reduce 优化](https://zhuanlan.zhihu.com/p/426978026)

的内容。希望后面能坚持下去。

欢迎大家关注哈:)