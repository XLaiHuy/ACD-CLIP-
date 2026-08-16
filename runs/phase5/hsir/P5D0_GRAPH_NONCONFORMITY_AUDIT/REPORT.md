# P5-D0 graph non-conformity leverage audit

Terminal: `GRAPH_AGGREGATION_LEVERAGE_INSUFFICIENT`.

The graph used all certified R0 relations before disjoint selection. Hodge potentials and S6 were materialized and hashed before GT was read. This is a post-hoc diagnostic; no candidate was implemented.

Certified edges: aligned=387594; shifted=1920878.
Hodge energy fractions: aligned gradient=1.000000000000001, residual=1.9785966175857728e-31; shifted gradient=1.0000000000000002, residual=2.3136112013797733e-17.
Native-grid edge mean Chebyshev distance: aligned=16.011307121765515; shifted=15.322728989830281.
S6 anomaly-image AUC: aligned=0.46816341763558644; shifted=0.552800005437179; class-bootstrap delta CI=[-0.10940568404513316, -0.048458427735159765].
Aligned graph positive-contamination fraction=0.06998958421905409; shifted=0.19272369107362688; old selected=0.03960581875762963; classes increased=12.
Gates: {'G0': True, 'G1': False, 'G2': False, 'G3': False, 'G4': True, 'G1_positive_mass_fraction': 0.06998958421905409, 'G1_old_fraction': 0.03960581875762963, 'G1_classes_increased': 12, 'G2_aligned_S6_auc': {'mean': 0.4903198176817319, 'ci95': [0.45926695652221294, 0.5248259853443948], 'n_classes': 12, 'unit': 'class', 'repetitions': 2000, 'seed': 7702}, 'G3_aligned_minus_shifted_auc': {'mean': -0.07733417816242534, 'ci95': [-0.10940568404513316, -0.048458427735159765], 'n_classes': 12, 'unit': 'class', 'repetitions': 2000, 'seed': 7703}, 'G4_mean_abs_classwise_spearman_S6_base': 0.08382103293679675}.

No D1 proposal is made unless the frozen supported terminal is reached.
