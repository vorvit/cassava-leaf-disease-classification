param(
  [string]$CudaIndex = "https://download.pytorch.org/whl/cu124"
)

# Installs CUDA-enabled PyTorch into the project venv without modifying uv.lock.
# This is intended for local development only (Task2 checks can stay CPU-only).
# IMPORTANT:
# - `uv run` syncs the venv by default and may overwrite CUDA torch back to CPU-only torch from `uv.lock`.
# - For GPU runs either:
#   - run commands via `.venv\\Scripts\\python.exe ...` / `.venv\\Scripts\\cassava.exe ...`, OR
#   - use `python -m uv run --no-sync ...` to keep the locally installed CUDA torch.

python -m uv pip uninstall --python .venv torch torchvision
python -m uv pip install --python .venv --extra-index-url $CudaIndex torch==2.6.0+cu124 torchvision==0.21.0+cu124

.\\.venv\\Scripts\\python.exe -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
