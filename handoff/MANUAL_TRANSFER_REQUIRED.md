# Optional large runtime artifacts

No P0 handoff item is blocked. The current scientific source/results,
context, Git history, and required Phase2B checkpoint are recoverable from
Git/Git LFS.

The following runtime artifacts still exist only on the rental filesystem and
were intentionally not copied into normal Git:

| Path | Size / files | Role | Status |
|---|---:|---|---|
| `/workspace/P5FR1_LATE_COMPLETION_SNAPSHOT` | 3.5G; 1,727 files; 3,721,908,151 file bytes | late-completion snapshot with 1,725 records plus status files | optional; committed scalar/results and hashes preserve the current conclusion |
| `/workspace/P5FR1C_ALL_CONFIG_EVIDENCE` | 131M; 1,726 files; 132,619,152 file bytes | all-config evidence cache | optional/reconstructible; scalar results and manifest are committed |
| `/tmp/p5fr1c_all_config_evidence` | 131M; 1,726 files; 132,619,152 file bytes | duplicate all-config evidence cache | optional duplicate |
| `/workspace/P5FR1C_DERIVE_INTERRUPTED_20260817T100730Z_640` | 49M; 641 files; 49,344,843 file bytes | interrupted partial derive | historical/debug only; not required |

Known committed all-config aggregate hash:
`3f6bc75c3c3a0eeed7a72b457d1e060a9aed55186e905b0ed1e9ad639be23b67`.

If these optional caches are needed later, preserve the rental filesystem or
transfer them with a resumable tool, for example:

```text
rsync -a --partial --info=progress2 <rental>:/workspace/P5FR1_LATE_COMPLETION_SNAPSHOT/ <new-machine>:/workspace/P5FR1_LATE_COMPLETION_SNAPSHOT/
```

No directory was deleted or modified by this handoff.
