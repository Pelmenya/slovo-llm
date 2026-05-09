# gpu-monitor.ps1 - Real-time GPU and container monitoring for slovo-llm
# Usage: .\gpu-monitor.ps1 [-ContainerName "ollama-laguna"]
# Shows GPU utilization, VRAM, power, and container stats in real-time

param(
    [string]$ContainerName = "ollama-laguna"
)

Write-Host "Monitoring GPU for container: $ContainerName" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "================================="

# Check if nvidia-smi available
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "Error: nvidia-smi not found. Is NVIDIA driver installed?" -ForegroundColor Red
    exit 1
}

# Get GPU name for reference
$gpuName = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
Write-Host "GPU: $gpuName" -ForegroundColor Green
Write-Host ""

try {
    while ($true) {
        # GPU stats
        $gpuStats = nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits 2>$null
        
        # Container stats
        $containerStats = docker stats --no-stream --format "{{.MemUsage}}" $ContainerName 2>$null
        if (-not $containerStats) { $containerStats = "down" }
        
        # Parse GPU stats
        $parts = $gpuStats -split ", "
        $vramUsed = $parts[0]
        $vramTotal = $parts[1]
        $gpuUtil = $parts[2]
        $power = $parts[3]
        
        # Calculate percentage
        if ($vramTotal -and $vramTotal -ne 0) {
            $vramPct = [math]::Round(($vramUsed / $vramTotal) * 100)
        } else {
            $vramPct = "N/A"
        }
        
        # Write status
        Write-Host ("GPU Util: {0,3}% | VRAM: {1,5}/{2,5} MiB ({3,3}%) | Power: {4,5}W | Container: {5}" -f `
            $gpuUtil, $vramUsed, $vramTotal, $vramPct, $power, $containerStats) -NoNewline
        Write-Host "`r" -NoNewline
        
        Start-Sleep -Seconds 2
    }
}
finally {
    Write-Host "`nMonitoring stopped." -ForegroundColor Cyan
}