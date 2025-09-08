# Reimplementation-of-FlashAttention-Triton-
This repo reimplements the logic of Flash Attention on a naive attention mechanism in 2 ways. First, we implement it with simple for-loop. Then we leverage Triton to optimize the parallelism with multiple SRAM - in other words, performing the online softmax and kernel fusion.
