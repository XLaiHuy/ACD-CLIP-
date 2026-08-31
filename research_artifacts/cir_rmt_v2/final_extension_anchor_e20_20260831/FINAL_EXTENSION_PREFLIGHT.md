# Final extension preflight

Status: PASS for the source-only E14-to-E20 continuation gate.

This record freezes the state observed before the extension. The continuation
uses the existing `last.pth` cursor and does not restart E1 or launch a parent
control run.

| field | verified value |
|---|---|
| branch | `research/cir-dfg-rmt-v2-signfix` |
| baseline HEAD | `617eccb26d869c72c46046f586cd957e4e7301ed` |
| remote branch HEAD | `617eccb26d869c72c46046f586cd957e4e7301ed` |
| architecture | `CIR_DFG_RMT_V2` v2 |
| config SHA256 | `064e8acd4369645f631030b5d60abf8615e878b50e9caff6a4a8b2439b64f81c` |
| architecture freeze SHA256 | `f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86` |
| source | VisA, seed 0, effective batch 6, FP32 |
| CLIP asset SHA256 | `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` |
| image anchor | Phase2B E14, lambda `0.001`, SHA256 `3eb6e2fe12f96b84745baf0f8a013f88c7f3a739283493a2ba5e31a35ad2f6c2` |
| existing resume cursor | `last.pth`, epoch 14, SHA256 `9b8fc5e7760037e772c9bd63d98ce56fcbbaa04f021258ca0d23aa8f2bf5ab81` |
| next epoch | 15 |
| E10 checkpoint SHA256 | `58af2ea6e3d92232498e3cb9bcf40b251e7116cfbac9a34d1abd4b07487aeaf0` |
| E12 checkpoint SHA256 | `bbbbfc6e24ac9dd1bfa87b596e3f6fe17a1b06cee6f5d4522ef67c4147a7e2f9` |
| E14 checkpoint SHA256 | `9b8fc5e7760037e772c9bd63d98ce56fcbbaa04f021258ca0d23aa8f2bf5ab81` |
| recorded training identity | `7d4ae4b8261aee8fa0d012188de1984296f7e7cf` |
| scheduler state at cursor | StepLR `last_epoch=14`, `_step_count=15`, gamma `0.9` |
| optimizer groups | image adapter, text adapter, soft prompt |
| optimizer state | Adam, betas `(0.9,0.999)`, eps `1e-8`, weight decay `0` |
| RNG state | CPU, CUDA, and dataloader generator present |
| E10/E12/E14 candidates | present and identity-valid |
| tracked diff at baseline | none |
| staged diff at baseline | none |
| unrelated worktree items | preserved untracked raw/forensic run roots and `.orig` files; not staged |

The extension tooling is an engineering-only continuation wrapper. It does
not alter the frozen config, architecture, loss, optimizer, scheduler, RMT
forward path, source sample, seed, CLIP asset, or checkpoint identity. No
Medical or MVTec evaluation is authorized by this preflight.
