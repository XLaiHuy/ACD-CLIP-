# Pre-solution identity audit

Status: PASS; this audit is a gate for bounded diagnostics.

| item | verified identity |
|---|---|
| branch | `research/cir-dfg-rmt-v2-signfix` |
| current Git HEAD | `0bed7b25d32945919a18cbea416a135c6111b806` |
| corrective scientific-code commit | `042174cdc63d9cb635566a1dae5b774056045383` |
| tracked change since corrective code commit | corrective archive additions only; no scientific code modification |
| architecture | `CIR_DFG_RMT_V2`, version 2 |
| architecture freeze SHA256 | `f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86` |
| parent config SHA256 | `d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb` |
| CIR config SHA256 | `064e8acd4369645f631030b5d60abf8615e878b50e9caff6a4a8b2439b64f81c` |
| CLIP asset SHA256 | `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` |
| VisA source identity SHA256 | `098f7cceef433c2e36992f75cca95a027e1601075de6ca04d0d385045aaae4a6` |
| seed / precision | seed `0`, FP32, AMP false, TF32 false |
| optimizer / scheduler | Adam; StepLR(step_size=1, gamma=0.9), restored and post-step at candidates |
| checkpoint identity | P and C0 E10/E12/E14/E16/E18/E20 exist and match the frozen corrective manifest hashes |
| P/C0 schema | image adapter, text adapter, and soft prompt state schemas are identical |
| baseline status | corrective training and P/C0/C05 source/Medical matrices completed before this objective |
| protected working-tree items | raw corrective/forensic directories and `.orig` files remain untracked and unstaged |

No checkpoint, config, architecture, evaluator, optimizer, loss, scheduler,
RMT, deployment operator, or frozen artifact was modified by this audit.
