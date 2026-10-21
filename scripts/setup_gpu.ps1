param(
    [string]$Python = 'python'
)

# 프로젝트 안의 격리된 환경만 구성한다. 기존 CPU 환경/드라이버는 변경하지 않는다.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$gpuEnvironment = Join-Path $projectRoot '.venv-gpu'
$gpuPython = Join-Path $gpuEnvironment 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $gpuPython)) {
    & $Python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12'"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
    & $Python -m venv $gpuEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the GPU environment.' }
}

& $gpuPython -c 'import sys; assert sys.prefix != sys.base_prefix; assert sys.version_info[:2] == (3, 12)'
if ($LASTEXITCODE -ne 0) { throw 'Invalid GPU environment.' }
& $gpuPython -m pip install --no-cache-dir -r (Join-Path $projectRoot 'requirements-gpu.txt')
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $gpuPython -m pip install --no-cache-dir -r (Join-Path $projectRoot 'requirements-training.txt')
if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
& $gpuPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
& $gpuPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda') + 1)"
if ($LASTEXITCODE -ne 0) { throw 'CUDA execution verification failed.' }
Write-Host "GPU environment ready: $gpuPython"
