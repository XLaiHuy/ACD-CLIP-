# C2 E10 checkpoint integrity incident

Status: disclosed and contained.

While preparing a temporary replay copy, /tmp/acdclip_phase2b_current_p_e10_historical/runs/phase2b_replay/adapter_10.pth was a symlink to the physical C2 E10 checkpoint. A serialization-only torch.save operation followed the symlink and replaced the original full payload.

- Original recorded SHA: 31ca8344c646693d0ee51941d39f28aa07b6a102c49d1efdc5e3cdf2ec8bcc50.
- Current observed SHA: ec4f472790241a9f746bb5b4ca6e31ca4782ec7333aab91ae691e9b2fb0c7347.
- Model tensor values: preserved in the stripped payload and replayable.
- Optimizer/scheduler/RNG metadata: no longer physically recoverable.
- Original metric rows and SHA: preserved in the previously frozen compact archives.
- Scientific consequence: C2 E10 is valid for model-state metric comparison, but not for exact full-checkpoint or resume-state verification.

No other C2 candidate checkpoint was intentionally modified. No retraining was launched to conceal or replace this incident.
