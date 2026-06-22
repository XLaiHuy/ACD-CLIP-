# ACD-CLIP++ Phase 2: Environment Setup & High-Speed Compilation Guide

This guide describes how to configure the environment on a new machine (particularly lab servers running Windows or Linux) to compile custom CUDA kernels for VMamba, ensuring maximum training and evaluation speed.

---

## 1. Quick Setup (Automated)

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repo_url>
   cd phase1_DFG+Attention/ACD-CLIP
   ```
2. Create and activate a Python virtual environment:
   * **Windows**:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * **Linux**:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
3. Run the interactive setup script:
   ```bash
   python setup_env.py
   ```
   * Select Option 1 (CUDA 12.1) to install PyTorch with GPU support.
   * The script will install base requirements and attempt to compile the high-speed `selective_scan` CUDA kernels.

---

## 2. Ensuring High-Speed Execution (GPU)

If the compilation fails, the code will fall back to native PyTorch scanning, which is mathematically correct but **significantly slower** due to a Python loop over the sequence length L=1369. To enable fast GPU training, follow the pre-requisite configurations below.

### 2.1 Match CUDA Toolkit with PyTorch CUDA Version
1. Check the PyTorch CUDA version inside your Python environment:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
   (Typically outputs `12.1` or `11.8`)
2. Download and install the matching **CUDA Toolkit** version from the official NVIDIA archive.
   * **Crucial Step**: In the NVIDIA installer, choose **Custom (Advanced)** and **uncheck the Driver option**. This prevents the installer from overwriting your existing, working display driver.

### 2.2 Install a C++ Compiler
* **Windows (Lab Servers)**:
  1. Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
  2. Select the **Desktop development with C++** workload.
  3. Ensure MSVC and Windows 10/11 SDK components are checked, then install.
* **Linux (Ubuntu)**:
  Install the compiler tools via apt:
  ```bash
  sudo apt update
  sudo apt install build-essential -y
  ```

### 2.3 Configure Environment Variables (PATH)
The compilation tools need to find the CUDA compiler (`nvcc`) and C++ compiler (`cl.exe` or `g++`).

#### On Linux (Ubuntu)
Add the following to your `~/.bashrc` (adjusting `cuda-12.1` to match your installed version):
```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```
Reload config: `source ~/.bashrc`.

#### On Windows (PowerShell/CMD)
1. Add the CUDA `bin` path to your System Environment variables (PATH):
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`
2. **Easiest way to build on Windows**:
   Instead of using a standard terminal, open the **x64 Native Tools Command Prompt for VS** (search for it in the Start menu). This console has all MSVC environment variables (`cl.exe`) pre-configured. Run your virtual environment activation and setup script inside this console:
   ```cmd
   .venv\Scripts\activate
   python setup_env.py
   ```

---

## 3. Training & Validation

Once setup is complete, verify the compilation and gradient flow:
```bash
python test_math_dynamic.py
```
If you see `SUCCESS: All dynamic adapter parameters have valid, non-zero gradients!`, you are ready to train:
```bash
python train.py --use_dynamic_conv
```
