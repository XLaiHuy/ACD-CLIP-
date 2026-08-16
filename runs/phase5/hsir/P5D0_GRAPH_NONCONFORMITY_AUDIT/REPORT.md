# P5-D0 graph non-conformity leverage audit

Terminal: `GRAPH_AGGREGATION_LEVERAGE_INSUFFICIENT`.

The primary graph used every certified R0 relation before disjoint selection. Edges use the frozen positive q_m flow; raw score gaps are diagnostic only. Hodge signals were materialized and hashed before GT was read. This is a post-hoc diagnostic; no candidate was implemented.

## 1. Did graph aggregation materially expand leverage?

Aligned positive-contamination mass touched fraction=0.050375630810926976; old selected fraction=0.03960581875762963; classes increased=12/12. Gate G1=FAIL.

## 2. Is the expansion grounded in aligned reference evidence?

Aligned S6 anomaly-image AUC=0.5519987230036029; shifted=0.5348643636601765; class-bootstrap aligned-minus-shifted CI=[-0.011155901701856366, 0.07483142857697955]. G2=PASS, G3=FAIL.

## 3. Does Hodge potential provide non-redundant anomaly information?

Mean absolute classwise Spearman(S6, base_m)=0.15301644054395885; G4=PASS. Pearson and Spearman associations with base, D_rank, and E_nonlocal are saved in DIAGNOSTIC_SIGNALS.json.

## 4. What failed/passed by class?

- candle: aligned S6 AUC=0.6501058099268949; shifted=0.4923225101556619; aligned positive mass=0.018910353145742412; classwise G3 direction=PASS.
- capsules: aligned S6 AUC=0.5041649632052331; shifted=0.5959758706763343; aligned positive mass=0.03831788049621195; classwise G3 direction=FAIL.
- cashew: aligned S6 AUC=0.5751717557251909; shifted=0.49914344428126123; aligned positive mass=0.07027564733398864; classwise G3 direction=PASS.
- chewinggum: aligned S6 AUC=0.571420495658466; shifted=0.5310191237880117; aligned positive mass=0.04501865527382522; classwise G3 direction=PASS.
- fryum: aligned S6 AUC=0.5219158535430087; shifted=0.48060646146885566; aligned positive mass=0.061206675163377376; classwise G3 direction=PASS.
- macaroni1: aligned S6 AUC=0.6229396257884692; shifted=0.6132068740768166; aligned positive mass=0.10634890481573787; classwise G3 direction=PASS.
- macaroni2: aligned S6 AUC=0.6662919092551697; shifted=0.6883890522213761; aligned positive mass=0.062004174427160524; classwise G3 direction=FAIL.
- pcb1: aligned S6 AUC=0.4723118279569892; shifted=0.5724899471074502; aligned positive mass=0.06702314722923178; classwise G3 direction=FAIL.
- pcb2: aligned S6 AUC=0.5979992411330005; shifted=0.5381579431732726; aligned positive mass=0.023863312050870868; classwise G3 direction=PASS.
- pcb3: aligned S6 AUC=0.6696865443425076; shifted=0.5531009288303294; aligned positive mass=0.010598234256247072; classwise G3 direction=PASS.
- pcb4: aligned S6 AUC=0.625858521918025; shifted=0.5052328250599687; aligned positive mass=0.015775045852041907; classwise G3 direction=PASS.
- pipe_fryum: aligned S6 AUC=0.5029956160542564; shifted=0.5218820059539123; aligned positive mass=0.13765969725961955; classwise G3 direction=FAIL.

## 5. Which exact terminal decision was reached?

`GRAPH_AGGREGATION_LEVERAGE_INSUFFICIENT`; candidate remains `NONE`. G0=True, G1=False, G2=True, G3=False, G4=True.

## 6. What single next research question follows?

If and only if the terminal is GRAPH_NONCONFORMITY_SUPPORTED_FOR_D1, can a bounded D1 use graph evidence without broad permutation or deployment mass relocation?

Graph energy fractions: aligned gradient=1.0, residual=1.4446556636000135e-31; shifted gradient=1.0, residual=2.2098421000308965e-31. Native-grid mean Chebyshev edge distance: aligned=16.011307121765515, shifted=15.322728989830281; SPATIAL_CONSTRAINT_REQUIRED=True.
