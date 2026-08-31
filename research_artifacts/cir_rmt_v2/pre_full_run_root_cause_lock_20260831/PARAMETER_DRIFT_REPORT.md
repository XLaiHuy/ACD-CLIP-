# Parameter drift by epoch

The CSV compares the frozen corrected parent checkpoint P with the corrected CIR checkpoint C0. Distances are descriptive; they are not by themselves evidence of harmful overspecialization.

No pre-update/init checkpoint was recoverable from the frozen run roots, so the requested common-initialization reference is explicitly unavailable.

| epoch | component | normalized L2 | flattened cosine | relative update |
|---:|---|---:|---:|---:|
| 10 | image_adapter | 0.725926 | 0.723578 | 0.766021 |
| 10 | text_adapter | 0.0769315 | 0.99705 | 0.0767547 |
| 10 | soft_prompt | 0.753802 | 0.720643 | 0.741551 |
| 12 | image_adapter | 0.739327 | 0.71367 | 0.778653 |
| 12 | text_adapter | 0.0805938 | 0.996763 | 0.0804151 |
| 12 | soft_prompt | 0.791175 | 0.691815 | 0.779344 |
| 14 | image_adapter | 0.749421 | 0.706882 | 0.785671 |
| 14 | text_adapter | 0.0836108 | 0.996515 | 0.0834267 |
| 14 | soft_prompt | 0.821039 | 0.667185 | 0.810905 |
| 16 | image_adapter | 0.756503 | 0.702443 | 0.789483 |
| 16 | text_adapter | 0.0860918 | 0.996305 | 0.0859022 |
| 16 | soft_prompt | 0.845857 | 0.645753 | 0.837715 |
| 18 | image_adapter | 0.762096 | 0.698531 | 0.793734 |
| 18 | text_adapter | 0.0881344 | 0.996128 | 0.087937 |
| 18 | soft_prompt | 0.864604 | 0.629253 | 0.857686 |
| 20 | image_adapter | 0.766236 | 0.696207 | 0.79512 |
| 20 | text_adapter | 0.0898649 | 0.995974 | 0.0896631 |
| 20 | soft_prompt | 0.880819 | 0.614867 | 0.874547 |
