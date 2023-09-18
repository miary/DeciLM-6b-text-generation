### DeciLM-6B Model

Create a python virtual environment

Install Cuda Toolkit: https://developer.nvidia.com/cuda-downloads
From command line, check if Cuda toolkit properly installed by running command: nvcc --version
Also, run command: nvidia-smi 

Install torch, torchvision, and torchaudio; follow instructions: https://pytorch.org/get-started/locally/
Note: If previous installation of torch, torchvision, torchaudio exists, run pip uninstall on all three packages

Run: pip install -r requirements.txt
Warning: If running on Windows platform, run bitsandbytes-windows instead of bitsandbytes