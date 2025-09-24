@echo off
echo ====================================
echo DN-DETR Windows Training Setup
echo ====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo Python found. Setting up virtual environment...

REM Create virtual environment
if exist "dn_detr_env" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q dn_detr_env
)

python -m venv dn_detr_env
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully.
echo.

REM Activate virtual environment
echo Activating virtual environment...
call dn_detr_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo Warning: Failed to install PyTorch with CUDA. Installing CPU version...
    pip install torch torchvision torchaudio
)

REM Install other dependencies
echo Installing other dependencies...
pip install cython scipy termcolor addict yapf timm psutil GPUtil submitit
if errorlevel 1 (
    echo Error: Failed to install basic dependencies
    pause
    exit /b 1
)

REM Install COCO API
echo Installing COCO API...
pip install pycocotools
if errorlevel 1 (
    echo Warning: Failed to install pycocotools from PyPI
    echo Trying to install from source...
    pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools"
)

REM Install Panoptic API
echo Installing Panoptic API...
pip install "git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi"

REM Check CUDA availability
echo.
echo Checking CUDA availability...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo.
echo ====================================
echo Setup completed successfully!
echo ====================================
echo.

REM Ask user for training configuration
set /p EPOCHS="Enter number of epochs (default 200): "
if "%EPOCHS%"=="" set EPOCHS=200

set /p PATIENCE="Enter early stopping patience (default 3): "
if "%PATIENCE%"=="" set PATIENCE=3

set /p BATCH_SIZE="Enter batch size (default 2): "
if "%BATCH_SIZE%"=="" set BATCH_SIZE=2

set /p DATASET_PATH="Enter dataset path (default coco2017_augmented): "
if "%DATASET_PATH%"=="" set DATASET_PATH=coco2017_augmented

echo.
echo Training Configuration:
echo - Epochs: %EPOCHS%
echo - Early Stopping Patience: %PATIENCE%
echo - Batch Size: %BATCH_SIZE%
echo - Dataset Path: %DATASET_PATH%
echo.

set /p CONFIRM="Start training now? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Training cancelled. You can start training later with train_windows.bat
    pause
    exit /b 0
)

echo.
echo ====================================
echo Starting DN-DETR Training...
echo ====================================
echo.

REM Start training
python train_custom.py -m dn_dab_detr --output_dir logs/dn_dab_detr/custom_training --batch_size %BATCH_SIZE% --epochs %EPOCHS% --lr_drop 150 --coco_path %DATASET_PATH%/ --use_dn --patience %PATIENCE% --num_workers 2 --save_checkpoint_interval 25

if errorlevel 1 (
    echo.
    echo Training failed or was interrupted.
    echo Check the error messages above.
) else (
    echo.
    echo Training completed successfully!
    echo Check the logs/dn_dab_detr/custom_training/ directory for results.
)

echo.
echo ====================================
echo Training session ended
echo ====================================
pause