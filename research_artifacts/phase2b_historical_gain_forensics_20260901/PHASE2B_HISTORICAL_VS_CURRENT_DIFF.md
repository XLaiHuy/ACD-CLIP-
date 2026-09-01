# Historical H2 versus current C2 differences

The CSV classifies every recovered difference. Core model dimensions, DFG settings, base image/text LRs, Adam defaults, batch size, and StepLR timing match. The meaningful differences are K-reg 0.002 to zero/stub, KG 0.01 to 0.001, AMP to FP32, soft-prompt LR 5e-5 to 1e-4, 15 to 20 epochs, historical versus current loader details, and pixel stride 4 versus 1 evaluation. These are multiple confounds, so the observed residual cannot be assigned to one cause.
