# Medical evaluator handoff

The exact Medical evaluator is maintained as a separate Git repository and is
not duplicated into this main repository.

- Repository: `https://github.com/XLaiHuy/ACD-CLIP-.git`
- Branch: `phase-fewshot-medical`
- Exact evaluator commit:
  `6bd932fbce0a425af5c8d3f7230dd7dc041568bd`
- Source checkout observed at:
  `/home/ai4/caohuy/ACD-CLIP-medical-test`
- Relevant entry points:
  `run_phase2cd_medical_eval.sh`, `phase2cd_medical_eval.py`,
  `requirements.txt`.

The exact commit is present on the remote `phase-fewshot-medical` branch.
Restore it on another machine with:

```bash
git clone -b phase-fewshot-medical \
  https://github.com/XLaiHuy/ACD-CLIP-.git ACD-CLIP-medical-test
cd ACD-CLIP-medical-test
git checkout 6bd932fbce0a425af5c8d3f7230dd7dc041568bd
```

Do not run the evaluator as part of archive restoration. The H2 target
evaluation results already committed in the main branch were produced with
this evaluator identity.
