@echo off
echo ====================================
echo DN-DETR Windows Training Script
echo ====================================
echo.

REM Check if virtual environment exists
if not exist "dn_detr_env\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup_and_train_windows.bat first to set up the environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call dn_detr_env\Scripts\activate.bat

REM Check if train_custom.py exists
if not exist "train_custom.py" (
    echo Error: train_custom.py not found!
    echo Make sure you're running this script from the DN-DETR directory.
    pause
    exit /b 1
)

REM Get training parameters from user
echo.
echo Training Configuration:
echo.

set /p EPOCHS="Enter number of epochs (default 200): "
if "%EPOCHS%"=="" set EPOCHS=200

set /p PATIENCE="Enter early stopping patience (default 3): "
if "%PATIENCE%"=="" set PATIENCE=3

set /p BATCH_SIZE="Enter batch size (default 2): "
if "%BATCH_SIZE%"=="" set BATCH_SIZE=2

set /p DATASET_PATH="Enter dataset path (default coco2017_augmented): "
if "%DATASET_PATH%"=="" set DATASET_PATH=coco2017_augmented

set /p OUTPUT_DIR="Enter output directory (default logs/dn_dab_detr/custom_training): "
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=logs/dn_dab_detr/custom_training

echo.
echo ====================================
echo Training Configuration Summary:
echo ====================================
echo Model: DN-DAB-DETR
echo Dataset: %DATASET_PATH%/
echo Epochs: %EPOCHS%
echo Early Stopping Patience: %PATIENCE%
echo Batch Size: %BATCH_SIZE%
echo Output Directory: %OUTPUT_DIR%
echo LR Drop: Epoch 150
echo GPU Monitoring: Enabled
echo ====================================
echo.

set /p CONFIRM="Start training with these settings? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Training cancelled.
    pause
    exit /b 0
)

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

echo.
echo ====================================
echo Starting DN-DETR Training...
echo ====================================
echo Dataset path: %DATASET_PATH%/
echo Epochs: %EPOCHS%
echo Patience: %PATIENCE%
echo Output directory: %OUTPUT_DIR%
echo GPU: Monitoring enabled
echo.

REM Start training with GPU monitoring
python train_custom.py ^
  -m dn_dab_detr ^
  --output_dir "%OUTPUT_DIR%" ^
  --batch_size %BATCH_SIZE% ^
  --epochs %EPOCHS% ^
  --lr_drop 150 ^
  --coco_path "%DATASET_PATH%/" ^
  --use_dn ^
  --patience %PATIENCE% ^
  --num_workers 2 ^
  --save_checkpoint_interval 25

REM Check training result
if errorlevel 1 (
    echo.
    echo ====================================
    echo Training Failed or Interrupted
    echo ====================================
    echo Check the error messages above for details.
    echo Common issues:
    echo - Dataset path incorrect
    echo - Insufficient GPU memory (try reducing batch_size)
    echo - Missing dependencies (run setup_and_train_windows.bat again)
    echo ====================================
) else (
    echo.
    echo ====================================
    echo Training Completed Successfully!
    echo ====================================
    echo Results saved in: %OUTPUT_DIR%
    echo.
    echo Files created:
    echo - checkpoint.pth (latest model)
    echo - log.txt (training logs)
    echo - eval/ (evaluation results)
    echo ====================================
)

echo.
echo Training session ended at: %date% %time%
echo Press any key to exit...
pause >nul