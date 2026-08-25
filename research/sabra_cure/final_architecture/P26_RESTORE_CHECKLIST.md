# P26 Restore Checklist

- Clone `XLaiHuy/ACD-CLIP-` with Git LFS installed.
- Fetch branches and tags.
- Check out `research/p26-sabra-cure-final-architecture-freeze-v1`.
- Confirm tag `sabra-final-p26-v1` resolves to the checked-out HEAD.
- Hydrate only the required LFS artifacts if they are pointer files.
- Run `bash scripts/restore_p26_sabra_final.sh`.
- Require `SABRA_FINAL_RESTORE_STATUS=READY`.
- Read `SABRA_FINAL.md` and `P26_AGENT_CONTEXT.md`.
- Do not access MVTec until a later explicit authorization.
- Never access Medical, tune the architecture, or substitute artifacts.
