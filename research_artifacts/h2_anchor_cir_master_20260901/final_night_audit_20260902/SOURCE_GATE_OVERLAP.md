# Source gate overlap audit

Status: `IN_DISTRIBUTION_ASSESSMENT`.

The 96-image source gate was compared by exact `image_path` against the VisA training manifest consumed by the H2 `TextAndImageDataset`. The loader reads all training-manifest rows and does not apply a category holdout filter.

| Quantity | Result |
|---|---:|
| Gate images | 96 |
| Training-manifest rows | 2162 |
| Exact image-path intersection | 96 |
| Overlap fraction | 1.000000 |
| Unmatched gate images | 0 |
| Gate categories present in training | 12 / 12 |

All gate images are therefore in the training distribution. The source result can assess behavior on sampled training-distribution images, but it is not a source holdout or a generalization validation. A clean future protocol should construct the source assessment split before training and keep it disjoint by exact image identity (and, if intended, by category).
