#!/bin/bash

echo "Stopping CARLA server..."
docker stop carla-server
docker rm carla-server
echo "✓ CARLA server stopped"
