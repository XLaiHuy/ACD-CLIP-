# H2 Extension Parity

Status: `PASS`

This fixed-input test compares the historical H2 native DFG and deployment path with the extension reconstruction using the common E0 model state. It is a portability/implementation test, not a training or target-selection result.

- H2 commit: `e03966997d4cecfd985943a4053a93e1e40197ec`
- E0 SHA256: `e3eb453a984ea4116d18e125905d75508257aaf01efad421b309819e9d5dda0f`
- Fixed batch: `candle/Data/Images/Anomaly/004.JPG, candle/Data/Images/Anomaly/011.JPG`
- Tolerance: `0.0001`
- DFG weight max abs diff: `0`
- Native-logit max abs diff: `0`
- Training-probability max abs diff: `0`
- Deployment-probability max abs diff: `0`
- Peer validity fraction: `1.000000`
- K-reg loss: `0.000463197648`
- Soft-prompt K-reg gradient norm: `2.2348936e-06`
- Detached W_K gradient norm: `0`

The K-reg check expects gradients through the soft-prompt input and no gradient through detached W_K, matching the historical implementation. The V2 direction check is synthetic and only verifies the configured abnormal-minus/normal-plus sign.
