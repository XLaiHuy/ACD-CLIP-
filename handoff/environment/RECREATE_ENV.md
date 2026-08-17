# Environment recreation

The source shell did not expose conda, micromamba, Python `torchhuy`, or a
project Python interpreter. This handoff therefore records the limitation
instead of fabricating a freeze. No model or checkpoint was loaded.

On the next machine, recreate the preferred environment before any research:

```bash
conda create -n torchhuy python=3.10 -y
conda activate torchhuy
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

The committed `requirements.txt` SHA256 is
`71751ee469d1d9d1e47ad51f4155a8d56558ff78fa66806dd4c374ef35e0dc3a`.
It requests PyTorch 2.5.1+cu121 and torchvision 0.20.1+cu121. Resolve the
appropriate PyTorch wheel/index for the next machine’s CUDA runtime, then
record a fresh `conda env export --from-history` and `pip freeze` before any
experiment.
