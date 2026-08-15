# Projection math review (not a frozen implementation)

Let `m_i` and `m_j` be native anomaly logits. If an accepted relation requires patch `i` to rank above `j`, define the strict base inversion gap `g = m_j - m_i`. Only `g > 0` is a strict inversion. At `g = 0`, no epsilon or tuned near-tie threshold is introduced.

## S: symmetric two-variable minimum projection

```text
minimize 0.5[(x_i-m_i)^2 + (x_j-m_j)^2]
subject to x_i >= x_j.
```

For `g > 0`, the KKT solution pools the pair:

```text
x_i = x_j = (m_i + m_j)/2
Delta_i = +g/2
Delta_j = -g/2.
```

For one isolated pair, unrelated patches are unchanged. The pair has `L1=g`, `L2=g/sqrt(2)`, and `Linf=g/2`. The negative update to `j` can move score mass away from a base-trusted patch. Reusing a patch across constraints creates a coupled projection problem and can cascade.

## P: positive-only minimum uplift

```text
minimize 0.5(x_i-m_i)^2
subject to x_i >= m_j,
       x_j = m_j.
```

For `g > 0`, `x_i=m_j`, `x_j=m_j`, so `Delta_i=+g` and `Delta_j=0`. This is one-sided and compatible with the nonnegative deployment operator, but its `L2` and `Linf` movement are twice S’s. It still requires a reliable pair proposal, abstention, and a per-patch authority limit. Repeated application is not bounded merely because each update is positive.

## G: broader partial-order projection

```text
minimize 0.5 ||x-m||_2^2
subject to x_i >= x_j for every accepted relation (i,j).
```

This standard constrained projection/isotonic-style problem can pool or move patches not in the initiating relation. A dense relation graph or full E sort approaches closed C1 behavior. Minimum distortion alone does not guarantee sparse participation, no cascade, limited spatial support, or deployment improvement.

## Freeze status

These equations review the plausible future families. None is frozen because Gate 3 cannot specify the GT-free relation proposal, E acceptance/abstention, conflict handling, and authority bound. No rank, score, spatial, or displacement threshold is selected or searched.
