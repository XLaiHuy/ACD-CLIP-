# K-reg forensics

H2 implements a genuine K-space regularizer. For a hybrid text embedding, it forms hard text features, the alpha-mixed main text features, and per-stage states. For each vision_text_k projection it computes normalized K-space features for the main and hard paths; the hard projection is detached, and the cosine loss is mean(1 - cosine(k_main, k_hard_detached)), averaged across stages, groups, and normal/abnormal channels. The detached W_K path prevents direct W_K updates while allowing the active soft/main contribution to receive the regularizer gradient.

H2 uses lambda_k=0.002 and includes lambda_k*k_loss in cls + seg + lambda_kg*kg + lambda_k*k. H2’s KG term is distinct: lambda_kg=0.01 aligns soft and hard text embeddings. H2 also logs per-stage K cosine statistics.

The current active trainer exposes lambda_k but _text_with_regularizers sets k_loss = torch.zeros((), device=device). C2 sets lambda_k=0.0, so no K-reg gradient is present. This is CURRENT_KREG_STATUS=STUB_ZERO and is best classified as an accidental feature removal relative to H2. The exact historical formula is recovered, but no restored-H2 training/parity experiment has yet isolated its causal contribution: KREG_CAUSAL_STATUS=ASSOCIATED_ONLY.
