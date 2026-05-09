#!/bin/bash
# gpu-monitor.sh - Real-time GPU and container monitoring for slovo-llm
# Usage: ./gpu-monitor.sh [container_name]
# Shows GPU utilization, VRAM, power, and container stats in real-time

set -e

CONTAINER=${1:-"ollama-laguna"}
echo "Monitoring GPU for container: $CONTAINER"
echo "Press Ctrl+C to stop"
echo "================================"

# Check if nvidia-smi available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi

# Get GPU name for reference
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown GPU")
echo "GPU: $GPU_NAME"
echo ""

while true; do
    # GPU stats
    GPU_STATS=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "N/A N/A N/A N/A")
    
    # Container stats
    CONTAINER_STATS=$(docker stats --no-stream --format "table {{.MemUsage}}" "$CONTAINER" 2>/dev/null || echo "N/A")
    
    # Parse GPU stats
    read -r VRAM_USED VRAM_TOTAL GPU_UTIL POWER <<< "$GPU_STATS"
    
    # Calculate percentage
    if [ "$VRAM_TOTAL" != "N/A" ] && [ -n "$VRAM_TOTAL" ]; then
        VRAM_PCT=$((VRAM_USED * 100 / VRAM_TOTAL))
    else
        VRAM_PCT="N/A"
    fi
    
    # Clear line and print
    printf "\rGPU Util: %3s%% | VRAM: %5s/%-5s MiB (%3s%%) | Power: %5sW | Container: %s" \
        "${GPU_UTIL:-N/A}" "$VRAM_USED" "$VRAM_TOTAL" "$VRAM_PCT" "${POWER:-N/A}" "${CONTAINER_STATS:-down}"
    
    sleep 2
done