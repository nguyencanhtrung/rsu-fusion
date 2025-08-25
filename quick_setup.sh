#!/bin/bash
set -e  # Exit on any error

echo " RSU Fusion Quick Setup for Ubuntu 20.04, 22.04 + NVIDIA"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu 20.04
check_os() {
    local ubuntu_version
    ubuntu_version=$(lsb_release -rs)
    if [[ "$ubuntu_version" != "20.04" && "$ubuntu_version" != "22.04" ]]; then
        print_error "This script is designed for Ubuntu 20.04 LTS or 22.04 LTS"
        exit 1
    fi
    print_status "Ubuntu $ubuntu_version LTS detected"
}

# Check NVIDIA GPU
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "NVIDIA driver not found. Please install NVIDIA drivers first."
        exit 1
    fi
    
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
    print_status "GPU detected: $gpu_info"
}

# Install Docker if not present
install_docker() {
    if ! command -v docker &> /dev/null; then
        print_status "Installing Docker..."
        sudo apt update
        sudo apt install -y docker.io docker-compose
        sudo usermod -aG docker $USER
        print_status "Docker installed. You may need to restart your session."
    else
        print_status "Docker already installed"
    fi
}

# Setup Anaconda environment
setup_conda() {
    if ! command -v conda &> /dev/null; then
        print_warning "Anaconda not found. Please install Anaconda first."
        print_status "Download from: https://www.anaconda.com/products/distribution"
        return 1
    fi
    
    print_status "Setting up Conda environment..."
    
    # Create environment
    if conda env list | grep -q "rsu-fusion"; then
        print_status "rsu-fusion environment already exists"
    else
        conda create -n rsu-fusion python=3.10 -y
        print_status "Created rsu-fusion environment"
    fi
    
    # Activate environment
    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate rsu-fusion
    
    print_status "Installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    print_status "Installing additional packages..."
    pip install opencv-python opencv-contrib-python
    pip install numpy scipy matplotlib
    pip install ultralytics
    pip install filterpy
    pip install torchreid
    pip install pyyaml tqdm loguru
    pip install carla==0.9.15
    
    print_status "Python packages installed"
}

# Setup CARLA Docker
setup_carla_docker() {
    print_status "Setting up CARLA Docker..."
    
    # Create CARLA directory
    mkdir -p ~/carla-docker
    cd ~/carla-docker
    
    # Create docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  carla:
    image: carlasim/carla:0.9.15
    ports:
      - "2000:2000"
      - "2001:2001"
      - "2002:2002"
    environment:
      - DISPLAY=$DISPLAY
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./data:/home/carla/data
    networks:
      - carla-network
    command: >
      bash -c "
      ./CarlaUE4.sh -RenderOffScreen -carla-world-port=2000 -carla-streaming-port=2001 -carla-secondary-port=2002
      "

networks:
  carla-network:
    driver: bridge
EOF
    
    mkdir -p data
    
    print_status "Pulling CARLA Docker image..."
    docker pull carlasim/carla:0.9.15
    
    print_status "CARLA Docker setup complete"
}

# Create project structure
create_project_structure() {
    print_status "Creating project structure..."
    
    mkdir -p rsu-fusion/{src,data,models,configs,logs,outputs}
    mkdir -p rsu-fusion/src/{detection,tracking,fusion,utils}
    mkdir -p rsu-fusion/data/{videos,calibration,test_data}
    mkdir -p rsu-fusion/models/{yolo,reid,tracking}
    
    print_status "Project structure created"
}

# Main setup function
main() {
    print_status "Starting RSU Fusion setup..."
    
    check_os
    check_gpu
    install_docker
    setup_conda
    setup_carla_docker
    create_project_structure
    
    print_status "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Restart your terminal session (for Docker group permissions)"
    echo "2. Activate conda environment: conda activate rsu-fusion"
    echo "3. Start CARLA: cd ~/carla-docker && docker-compose up -d"
    echo "4. Run verification: python test_setup.py"
    echo "5. Test CARLA Town 10: python test_carla_town10.py"
}

# Run main function
main "$@"
