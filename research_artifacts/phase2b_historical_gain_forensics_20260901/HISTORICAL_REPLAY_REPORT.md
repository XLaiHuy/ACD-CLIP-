# Historical replay report

Status: PASS for exact H2 replay. The historical evaluator at source hash 7bdd8cc6ada90467285a79ced9599ed778c6dc2a0ba6596d2f3311fa637fae9d was run in the detached H2 source worktree against checkpoint ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb, with batch 8, six workers, metric thresholds unset, and pixel stride 4. The six-domain macro is 90.9750 Pixel AUROC / 40.3483 Pixel AP, matching the logged 90.98 / 40.35 after the historical two-decimal reporting.

Status: PASS for H2 same-checkpoint current-evaluator replay. The current exact evaluator hash cbcfaf4b2eda645fc6b440ed9bb486b5fb6b6f3af908e1c0ec70bafe13db0797 loaded the same H2 model-state checkpoint through the explicitly recorded legacy metadata bypass. The six-domain macro is 90.9222 / 40.3731.

H1 E9 was also replayed through the current evaluator; its six-domain pixel macro is 90.7119 / 39.8490. C2 P E10 current-evaluator values were read from the frozen corrective matrix, with the original recorded checkpoint SHA preserved. H2/H1 historical selection is retrospective Medical-informed and is not target-blind.
