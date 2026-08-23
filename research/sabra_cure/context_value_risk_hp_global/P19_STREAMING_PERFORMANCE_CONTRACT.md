# P19 Streaming and Performance Contract

Workers use the accepted exact grouped-delta AP engine only. Image target
construction computes pAP only; final comparator evaluation computes pAUROC
once from already constructed SAFE20/E40 maps. Per source class, maps are
current-class RAM only or exact fold-local `.npy` cache; no multi-source RAM
cache and no cross-fold cache exists.

Source selection makes one cache load per source class: all five frozen q
candidates are evaluated while that cache is active. Candidate-q results and
selection use P14's exact rule. The held class constructs SAFE20/E40 once and
reuses them for all required comparators.

Engineering GO requires AP parity error 0.0, speedup >=5x, selected worker
parity exact, projected science <=3.5 h, full science+audit <=6.5 h, and
global metric+audit <=10 min. No performance mechanism may alter science.
