#!/bin/bash

# DN-DETR One-Click Deployment Script
# Supports: Environment setup, dependency installation, training, and inference

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="DN-DETR"
VENV_NAME="dn_detr_env"
PYTHON_VERSION="python3"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_system_requirements() {
    print_status "Checking system requirements..."

    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.7 or higher."
        exit 1
    fi

    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $python_version found"

    # Check pip
    if ! command_exists pip3; then
        print_error "pip3 is not installed. Please install pip3."
        exit 1
    fi

    # Check CUDA availability
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        print_warning "No NVIDIA GPU detected. Training will use CPU (very slow)."
    fi

    # Check available disk space (minimum 20GB recommended)
    available_space=$(df . | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 20971520 ]; then  # 20GB in KB
        print_warning "Less than 20GB disk space available. Consider freeing up space."
    fi
}

# Function to setup virtual environment
setup_environment() {
    print_status "Setting up virtual environment..."

    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment already exists. Removing and recreating..."
        rm -rf "$VENV_NAME"
    fi

    $PYTHON_VERSION -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    print_success "Virtual environment created and activated"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."

    # Ensure virtual environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source "$VENV_NAME/bin/activate"
    fi

    # Install PyTorch first (with CUDA support if available)
    if command_exists nvidia-smi; then
        print_status "Installing PyTorch with CUDA support..."
        pip install torch>=1.5.0 torchvision>=0.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch (CPU-only)..."
        pip install torch>=1.5.0 torchvision>=0.6.0 torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install other requirements
    pip install -r requirements.txt

    print_success "Dependencies installed successfully"
}

# Function to validate dataset path
validate_dataset() {
    local dataset_path="$1"

    if [ -z "$dataset_path" ]; then
        print_error "Dataset path is required"
        echo "Usage: $0 train <dataset_path>"
        echo "Example: $0 train /path/to/your/dataset"
        exit 1
    fi

    if [ ! -d "$dataset_path" ]; then
        print_error "Dataset directory does not exist: $dataset_path"
        exit 1
    fi

    # Check for common dataset structure
    if [ ! -d "$dataset_path/train2017" ] && [ ! -d "$dataset_path/images" ]; then
        print_warning "Dataset structure may not be standard. Expected 'train2017' or 'images' directory."
        print_warning "Continuing anyway..."
    fi

    if [ ! -d "$dataset_path/annotations" ]; then
        print_warning "No 'annotations' directory found. Make sure annotation files exist."
    fi

    print_success "Dataset path validated: $dataset_path"
}

# Function to run training
run_training() {
    local dataset_path="$1"

    print_status "Starting training with dataset: $dataset_path"

    # Validate dataset path
    validate_dataset "$dataset_path"

    # Ensure virtual environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source "$VENV_NAME/bin/activate"
    fi

    # Check if custom training script exists
    if [ -f "train_custom.py" ]; then
        print_status "Using custom training script..."
        python3 train_custom.py \
            -m dn_dab_detr \
            --output_dir logs/dn_dab_detr/training \
            --batch_size 2 \
            --epochs 200 \
            --lr_drop 150 \
            --coco_path "$dataset_path" \
            --use_dn \
            --patience 3 \
            --num_workers 2 \
            --save_checkpoint_interval 25
    else
        print_status "Using default training script..."
        python3 main.py \
            -m dn_dab_detr \
            --output_dir logs/dn_dab_detr/training \
            --batch_size 2 \
            --epochs 200 \
            --lr_drop 150 \
            --coco_path "$dataset_path" \
            --use_dn
    fi

    print_success "Training completed!"
}

# Function to run inference
run_inference() {
    print_status "Running inference..."

    # Ensure virtual environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source "$VENV_NAME/bin/activate"
    fi

    if [ -f "inference.py" ]; then
        python3 inference.py "$@"
        print_success "Inference completed!"
    else
        print_error "inference.py not found. Please check the project structure."
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up..."

    # Remove temporary files
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup                    - Setup environment and install dependencies"
    echo "  train <dataset_path>     - Start training with specified dataset"
    echo "  inference                - Run inference (requires model checkpoint)"
    echo "  clean                    - Clean up temporary files"
    echo "  status                   - Show system status and requirements"
    echo ""
    echo "Examples:"
    echo "  $0 setup                           # Setup environment only"
    echo "  $0 train /path/to/your/dataset     # Train with custom dataset"
    echo "  $0 train ./my_dataset              # Train with local dataset"
    echo "  $0 inference --help                # Show inference options"
}

# Function to show system status
show_status() {
    print_status "System Status:"
    echo "Project: $PROJECT_NAME"
    echo "Virtual Environment: $VENV_NAME"
    echo "Python: $(python3 --version 2>&1)"

    if [ -d "$VENV_NAME" ]; then
        echo -e "Environment Status: ${GREEN}Ready${NC}"
    else
        echo -e "Environment Status: ${RED}Not Setup${NC}"
    fi

    if command_exists nvidia-smi; then
        echo -e "GPU Support: ${GREEN}Available${NC}"
        nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits | head -1
    else
        echo -e "GPU Support: ${YELLOW}Not Available${NC}"
    fi

    # Dataset status is not shown since it's manually specified
}

# Main execution logic
main() {
    echo "======================================="
    echo "     DN-DETR One-Click Deployment"
    echo "======================================="
    echo ""

    case "${1:-}" in
        "setup")
            check_system_requirements
            setup_environment
            install_dependencies
            ;;
        "train")
            shift
            run_training "$1"
            ;;
        "inference")
            shift
            run_inference "$@"
            ;;
        "clean")
            cleanup
            ;;
        "status")
            show_status
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        "")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Trap to handle script interruption
trap 'print_error "Script interrupted"; exit 1' INT

# Run main function with all arguments
main "$@"