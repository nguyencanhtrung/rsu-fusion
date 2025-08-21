# RSU Digital Twin - Environment Setup Guide

This guide will help you set up the development environment for the RSU Digital Twin project, which implements a multi-camera fusion tracker for V2X applications.

## Prerequisites

- **Operating System**: Ubuntu 20.04 LTS
- **Python**: Version 3.10 or higher
- **GPU**: NVIDIA 3060Ti (8GB VRAM) with CUDA support
- **Memory**: At least 16GB RAM
- **Storage**: At least 20GB free space for dependencies and models
- **Docker**: Docker and Docker Compose (for CARLA)

## 1. Python Environment Setup (Anaconda)

1. **Install Anaconda** (if not already installed):
   ```bash
   # Download and install Anaconda for Ubuntu 20.04
   wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
   bash Anaconda3-2024.06-Linux-x86_64.sh
   
   # Restart terminal or run:
   source ~/.bashrc
   ```

2. **Create a new environment**:
   ```bash
   conda create -n rsu-fusion python=3.10
   conda activate rsu-fusion
   ```

## 2. Core Dependencies Installation

### 2.1 PyTorch Installation (for NVIDIA 3060Ti)

**Check NVIDIA driver and CUDA compatibility**:
```bash
# Check your NVIDIA driver version
nvidia-smi

# Check available CUDA versions
ls /usr/local/cuda*/version.txt 2>/dev/null || echo "CUDA not found in standard location"
```

**Install PyTorch with CUDA 12.1 (compatible with 3060Ti)**:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Verify GPU detection**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 2.2 Computer Vision Libraries

```bash
# OpenCV for image processing
pip install opencv-python opencv-contrib-python

# Core numerical computing
pip install numpy scipy

# Image processing and augmentation
pip install Pillow albumentations
```

### 2.3 Object Detection (YOLOv8)

```bash
# Ultralytics YOLOv8
pip install ultralytics

# Download pre-trained weights (will be done automatically on first use)
# But you can pre-download them:
# python -c "from ultralytics import YOLO; model = YOLO('yolov8m.pt')"
```

### 2.4 Tracking Libraries

```bash
# For Kalman filtering
pip install filterpy

# For Hungarian algorithm (assignment problem)
pip install scipy

# For appearance-based Re-ID models
pip install torchreid
```

### 2.5 Utility Libraries

```bash
# For configuration management
pip install pyyaml

# For logging and progress bars
pip install tqdm loguru

# For data serialization
pip install pickle-mixin

# For mathematical operations
pip install scikit-learn

# For plotting and visualization
pip install matplotlib seaborn plotly
```

## 3. CARLA Simulator Setup (Docker)

### 3.1 Docker Installation

1. **Install Docker and Docker Compose**:
   ```bash
   # Update package list
   sudo apt update
   
   # Install Docker
   sudo apt install docker.io docker-compose
   
   # Add user to docker group (to run without sudo)
   sudo usermod -aG docker $USER
   
   # Restart to apply group changes
   newgrp docker
   ```

2. **Verify Docker installation**:
   ```bash
   docker --version
   docker-compose --version
   ```

### 3.2 CARLA Docker Setup

1. **Pull CARLA 0.9.15 Docker image**:
   ```bash
   docker pull carlasim/carla:0.9.15
   ```

2. **Create CARLA Docker setup**:
   ```bash
   # Create CARLA directory
   mkdir -p ~/carla-docker
   cd ~/carla-docker
   
   # Create docker-compose.yml file
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
   
   # Create data directory for sharing files
   mkdir -p data
   ```

3. **Start CARLA Docker container**:
   ```bash
   # Allow X11 forwarding (for GUI if needed)
   xhost +local:docker
   
   # Start CARLA
   docker-compose up -d
   
   # Check if CARLA is running
   docker-compose logs carla
   ```

### 3.3 CARLA Python API Installation

1. **Install CARLA Python package**:
   ```bash
   # Make sure you're in your conda environment
   conda activate rsu-fusion
   
   # Install CARLA Python API
   pip install carla==0.9.15
   ```

### 3.4 Town 10 Configuration

1. **Test CARLA connection and load Town 10**:
   ```bash
   # Create a test script
   cat > test_carla_town10.py << 'EOF'
   import carla
   import time
   
   def test_carla_town10():
       try:
           # Connect to CARLA
           client = carla.Client('localhost', 2000)
           client.set_timeout(10.0)
           
           print(f"CARLA Server Version: {client.get_server_version()}")
           
           # Load Town 10
           world = client.load_world('Town10HD')  # or 'Town10' for regular version
           print(f"Loaded world: {world.get_map().name}")
           
           # Get basic world info
           settings = world.get_settings()
           print(f"Synchronous mode: {settings.synchronous_mode}")
           print(f"Fixed delta seconds: {settings.fixed_delta_seconds}")
           
           # List available spawn points
           spawn_points = world.get_map().get_spawn_points()
           print(f"Available spawn points: {len(spawn_points)}")
           
           print("✅ CARLA Town 10 setup successful!")
           
       except Exception as e:
           print(f"❌ CARLA connection failed: {e}")
           return False
       
       return True
   
   if __name__ == "__main__":
       test_carla_town10()
   EOF
   
   # Run the test
   python test_carla_town10.py
   ```

## 4. Development Environment Setup

### 4.1 VS Code Configuration

**Recommended VS Code extensions** (install via terminal or Extension Marketplace):
```bash
# Python development
code --install-extension ms-python.python
code --install-extension ms-python.pylint
code --install-extension ms-python.black-formatter

# Jupyter notebooks
code --install-extension ms-toolsai.jupyter

# Docker support
code --install-extension ms-azuretools.vscode-docker

# Git and version control
code --install-extension eamodio.gitlens

# YAML/JSON support
code --install-extension redhat.vscode-yaml
```

**VS Code workspace settings** (create `.vscode/settings.json`):
```bash
# Create VS Code settings directory
mkdir -p .vscode

cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "~/anaconda3/envs/rsu-fusion/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.formatting.provider": "black",
    "python.linting.pylintEnabled": true,
    "python.linting.enabled": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/node_modules": true
    },
    "docker.defaultPlatform": "linux"
}
EOF
```

### 4.2 Project Structure Creation

```bash
# Create project directories
mkdir -p rsu-fusion/{src,data,models,configs,logs,outputs}
mkdir -p rsu-fusion/src/{detection,tracking,fusion,utils}
mkdir -p rsu-fusion/data/{videos,calibration,test_data}
mkdir -p rsu-fusion/models/{yolo,reid,tracking}
```

## 5. Additional Dependencies for Advanced Features

### 5.1 Re-ID Models (OSNet)

```bash
# Install additional dependencies for Re-ID
pip install gdown  # For downloading models from Google Drive

# Install torchreid for OSNet models
pip install torchreid
```

### 5.2 SAE J2735 Support

```bash
# For ASN.1 encoding/decoding (stretch goal)
pip install pyasn1 pyasn1-modules

# For JSON handling
pip install jsonschema
```

### 5.3 Visualization Tools

```bash
# For advanced plotting
pip install plotly dash

# For 3D visualization
pip install open3d

# For video processing
pip install moviepy
```

## 6. Verification

### 6.1 Environment Verification

Run the provided verification script:
```bash
# Make sure you're in your conda environment
conda activate rsu-fusion

# Run the verification script
python test_setup.py
```

### 6.2 CARLA Docker Verification

Test CARLA Docker setup:
```bash
# Test CARLA Town 10 setup (from section 3.4)
python test_carla_town10.py

# Check CARLA Docker container status
docker-compose ps

# Verify CARLA is accessible
python -c "
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
print(f'CARLA version: {client.get_server_version()}')
world = client.get_world()
print(f'Current world: {world.get_map().name}')
"
```

### 6.3 GPU and CUDA Verification

```bash
# Check GPU detection in Python
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

## 7. CARLA Docker Management

### 7.1 Docker Commands

**Start CARLA**:
```bash
cd ~/carla-docker
docker-compose up -d
```

**Stop CARLA**:
```bash
docker-compose down
```

**View CARLA logs**:
```bash
docker-compose logs -f carla
```

**Restart CARLA**:
```bash
docker-compose restart carla
```

### 7.2 Performance Optimization for NVIDIA 3060Ti

**Monitor GPU usage**:
```bash
# Install nvidia-htop for better GPU monitoring
pip install nvidia-htop

# Monitor GPU usage
nvidia-htop
# or
watch -n 1 nvidia-smi
```

**CARLA Docker resource limits** (add to docker-compose.yml if needed):
```yaml
services:
  carla:
    # ... existing config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 6G  # Adjust based on your 8GB VRAM
```

## 8. Troubleshooting

### Common Issues:

1. **CUDA Issues with NVIDIA 3060Ti**:
   - Verify NVIDIA drivers: `nvidia-smi`
   - Check CUDA version compatibility: `nvcc --version`
   - For 3060Ti, ensure you have driver version 470+ 
   - Reinstall PyTorch with correct CUDA version if needed

2. **CARLA Docker Issues**:
   - Ensure Docker daemon is running: `sudo systemctl status docker`
   - Check if ports are available: `netstat -tuln | grep 2000`
   - Verify X11 forwarding: `xhost +local:docker`
   - Check Docker logs: `docker-compose logs carla`

3. **Memory Issues (8GB VRAM)**:
   - Reduce batch sizes in deep learning models
   - Use mixed precision training: `torch.cuda.amp`
   - Monitor GPU memory: `nvidia-smi` or `nvidia-htop`

4. **Anaconda Environment Issues**:
   - Activate environment: `conda activate rsu-fusion`
   - Check environment: `conda env list`
   - Verify packages: `conda list`

5. **Import Errors**:
   - Ensure you're in the correct conda environment
   - Check if packages are installed: `pip list | grep <package>`
   - Reinstall problematic packages: `pip install --upgrade <package>`

### Getting Help:

If you encounter issues during setup, please provide:
- Ubuntu 20.04 LTS system information: `lsb_release -a`
- Python version: `python --version`
- Conda environment info: `conda info --envs`
- NVIDIA driver info: `nvidia-smi`
- Docker status: `docker --version && docker-compose --version`
- Error messages (full traceback)
- Output of the verification script

## 9. Next Steps

Once your environment is set up:

1. Run the verification script to ensure everything is working
2. Proceed to **Day 1** of the implementation sprint
3. Start with the single-camera 2D tracking implementation

---

**Note**: This setup guide assumes you're following the 7-day implementation plan outlined in the project README. Each day's tasks will build upon this foundation.
