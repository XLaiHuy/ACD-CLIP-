import os
import sys
import subprocess

def run_cmd(cmd, description=None):
    if description:
        print(f"\n>>> {description}...")
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def main():
    print("=========================================================================")
    print("               ACD-CLIP++ Phase 2 Environment Setup Script               ")
    print("=========================================================================")
    
    # 1. Ask for PyTorch installation choice
    print("\nSelect PyTorch Installation option:")
    print("1) Install PyTorch with CUDA 12.1 (Recommended for GPU servers/labs)")
    print("2) Install PyTorch for CPU only (For local laptop testing)")
    print("3) Skip PyTorch installation (Already installed)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        run_cmd("pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121", "Installing PyTorch with CUDA 12.1")
    elif choice == '2':
        run_cmd("pip install torch torchvision", "Installing PyTorch (CPU)")
    else:
        print("\n>>> Skipping PyTorch installation.")
        
    # 2. Install base requirements
    run_cmd("pip install -r requirements.txt", "Installing other requirements from requirements.txt")
    
    # 3. Attempt to compile selective_scan kernels for VMamba
    print("\n>>> Attempting to compile VMamba selective_scan kernels...")
    print("Note: This step requires CUDA Toolkit and C++ compilation tools (e.g. Visual Studio Build Tools).")
    
    success = run_cmd("pip install -e ./VMamba/kernels/selective_scan", "Compiling selective_scan kernels")
    
    if success:
        print("\n[SUCCESS] VMamba selective_scan kernels compiled and installed successfully!")
        print("VMamba will run at native high speed using compiled CUDA kernels.")
    else:
        print("\n=========================================================================")
        print("  WARNING: SELECTIVE SCAN COMPILATION FAILED (FALLBACK MODE ENABLED)   ")
        print("=========================================================================")
        print("The installation of the custom selective_scan CUDA kernels failed.")
        print("The code will fall back to native PyTorch scanning, but please note:")
        print("--> HUY NOTE: RUNNING Huấn luyện/Test TRONG CHẾ ĐỘ FALLBACK SẼ CỰC KỲ CHẬM!")
        print("Do sử dụng vòng lặp Python thuần theo chiều dọc của sequence length L=1369.")
        print("\nĐể biên dịch thành công chạy tốc độ cao trên máy lab trường:")
        print("1. Hãy chắc chắn máy lab đã cài đặt CUDA Toolkit (trùng với phiên bản PyTorch).")
        print("2. Đảm bảo có sẵn trình biên dịch C++:")
        print("   - Trên Linux (Lab Ubuntu): Chạy 'sudo apt install build-essential'")
        print("   - Trên Windows: Cài đặt Visual Studio Build Tools và tích chọn C++ workload.")
        print("3. Đảm bảo các biến môi trường PATH đã trỏ đúng đến 'nvcc' và 'cl' / 'g++'.")
        print("\nBạn vẫn có thể chạy thử nghiệm nhỏ để test lỗi logic, nhưng hãy sửa lỗi")
        print("biên dịch trên trước khi chạy huấn luyện chính thức.")
        print("=========================================================================")

    # 4. Run sanity validation test
    print("\nWould you like to run the mathematical sanity check validation now?")
    val_choice = input("Run validation? (y/n): ").strip().lower()
    if val_choice == 'y':
        run_cmd("python test_math_dynamic.py", "Running math and gradient check")
        
    print("\n=========================================================================")
    print("Setup completed! To start training on VisA with dynamic conv, use:")
    print("python train.py --use_dynamic_conv")
    print("=========================================================================")

if __name__ == "__main__":
    main()
