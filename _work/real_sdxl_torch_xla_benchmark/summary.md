# Real SDXL Torch/XLA TPU Benchmark

- Model: `KBlueLeaf/Kohaku-XL-Zeta`
- Backend: `torch_xla`
- Device target: `TPU v5e-8`
- Resolution: `1024x1024`
- Steps: `25`
- Output: full image `pil`
- Prompt: `masterpiece, best quality, cinematic fantasy character portrait, detailed lighting`

| num_images_per_prompt | status | cold latency sec | warm median sec | images/sec | XLA compile evidence | peak memory/error |
|---:|---|---:|---:|---:|---|---|
| 1 | success | 363.6499 | 9.8738 | 0.1013 | cold CompileTime samples=5; cold Uncached=5; warm Cached=29; warm ExecuteReplicated=29 |  |
| 8 | success | 391.4695 | 21.6474 | 0.3696 | cold CompileTime samples=4; cold Uncached=3; warm Cached=29; warm ExecuteReplicated=29 |  |
| 32 | failed |  |  |  | cold CompileTime samples=None; cold Uncached=None; warm Cached=None; warm ExecuteReplicated=None | ValueError: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 16.29G of 15.75G hbm. Exceeded hbm capacity by 558.95M. |
| 128 | failed |  |  |  | cold CompileTime samples=None; cold Uncached=None; warm Cached=None; warm ExecuteReplicated=None | ValueError: Allocation (size=21474836480) would exceed memory (size=17179869184) :: #allocation14167 [shape = 'f32[320,4096,4096]{1,2,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.30572.remat@{1}'] :: <no-hlo-instruction> |
