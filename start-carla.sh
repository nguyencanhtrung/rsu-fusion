#!/bin/bash

# Stop any existing CARLA container
docker stop carla-server 2>/dev/null || true
docker rm carla-server 2>/dev/null || true

# Start CARLA server with GPU support
echo "Starting CARLA server with GPU support..."
docker run -d \
  --name carla-server \
  -p 2000:2000 \
  -p 2001:2001 \
  -p 2002:2002 \
  --gpus all \
  -v $(pwd)/data:/home/carla/data \
  carlasim/carla:0.9.15 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen

echo "CARLA server starting..."
sleep 5

# Check status
if docker ps | grep -q carla-server; then
    echo "✓ CARLA server is running"
    echo " Available on localhost:2000"
    echo " Check logs with: docker logs carla-server -f"
else
    echo "𐄂 CARLA server failed to start"
    echo " Check logs with: docker logs carla-server"
fi
