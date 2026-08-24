# P25 GT-Free Feature Contract

The benefit selector has exactly 32 features in this fixed order. F01--F22
are the frozen R2-v2 `HARM_ORDER`, including its inherited 14 direction/trust
features and eight derived direction/uncertainty/harm features. P25 adds:

23. continuous harm risk
24. frozen harm-policy action indicator
25. affected-support native-score-rank median
26. affected-support native-score-rank q90
27. mean signed candidate score delta / robust within-image score scale
28. q90 absolute candidate score delta / robust within-image score scale
29. median signed empirical rank shift over affected support
30. q90 absolute empirical rank shift
31. top-5-percent rank-boundary crossing fraction
32. top-20-percent rank-boundary crossing fraction

All F23--F32 derive solely from frozen GT-free native scores, frozen direction,
harm predictions, and the known deployment operator. Labels/masks/V are never
an input to deployed features. There is no feature search, embedding, CNN,
Transformer, MLP, or tree ensemble.
