# H2 functional feature anchor bounded retry R1

R1 preserves invalid run `84404c9` and repeats only its stage-2/3 token-cosine
mechanism. It restores RNG after candidate teacher creation, enforces exact
augmented-batch hashes, caps the functional image-side gradient at `rho=.10`,
and uses the preregistered 500-attempt numerical rule. No target inference,
tuning, full E15, or second retry is authorized.
