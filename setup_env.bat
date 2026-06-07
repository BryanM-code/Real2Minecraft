@echo off
REM Real2Minecraft v2 — Environment Setup
REM Run this once to create / update the r2m_v2 venv.
REM Requires Python 3.10 installed (py launcher must see it).
REM ============================================================

set PROJECT=%~dp0
set PYTHON=%PROJECT%r2m_v2\Scripts\python.exe
set PIP=%PROJECT%r2m_v2\Scripts\pip.exe

echo.
echo ============================================================
echo  Real2Minecraft v2 — Environment Setup
echo ============================================================
echo.

REM ── 1. Create venv if needed ─────────────────────────────────
if not exist "%PYTHON%" (
    echo [1/6] Creating r2m_v2 venv with Python 3.10 ...
    py -3.10 -m venv "%PROJECT%r2m_v2"
    if errorlevel 1 (
        echo ERROR: Could not create venv. Make sure Python 3.10 is installed.
        pause & exit /b 1
    )
) else (
    echo [1/6] r2m_v2 venv already exists, skipping creation.
)

REM ── 2. Upgrade pip ───────────────────────────────────────────
echo [2/6] Upgrading pip ...
"%PYTHON%" -m pip install --upgrade pip

REM ── 3. PyTorch 2.1.0 + CUDA 12.1 ────────────────────────────
echo [3/6] Installing PyTorch 2.1.0 + CUDA 12.1 (this is ~2 GB) ...
"%PIP%" install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 ( echo ERROR: torch install failed. & pause & exit /b 1 )

REM ── 4. xformers ──────────────────────────────────────────────
echo [4/6] Installing xformers ...
"%PIP%" install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121

REM Windows Triton workaround
"%PIP%" install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl

REM ── 5. InstantMesh requirements ───────────────────────────────
echo [5/6] Installing InstantMesh requirements ...
"%PIP%" install pytorch-lightning==2.1.2
"%PIP%" install transformers==4.34.1
"%PIP%" install diffusers==0.20.2
"%PIP%" install einops omegaconf torchmetrics accelerate tensorboard
"%PIP%" install huggingface-hub imageio[ffmpeg] plyfile
"%PIP%" install PyMCubes xatlas
"%PIP%" install rembg
REM nvdiffrast — pre-built wheel if available, otherwise build from source
"%PIP%" install nvdiffrast
if errorlevel 1 (
    echo NOTE: nvdiffrast pip install failed.
    echo Trying from source (needs CUDA toolkit + MSVC^) ...
    "%PIP%" install git+https://github.com/NVlabs/nvdiffrast/
)

REM ── 6. App UI dependencies + numpy pin ───────────────────────
echo [6/6] Installing app UI dependencies ...
"%PIP%" install trimesh matplotlib pillow
REM Pin numpy<2 last — pytorch-lightning may re-pull numpy 2.x
"%PIP%" install "numpy<2"

echo.
echo ============================================================
echo  Setup complete!
echo  Launch the app with:  r2m_v2\Scripts\python.exe app.py
echo ============================================================
pause
