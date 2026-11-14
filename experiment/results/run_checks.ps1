# Remaining protocol checks, one after another (never two GPU jobs at once):
#  1. train-on-synthetic, test-on-real (main CNN, 3 scenarios x 3 levels x 3 seeds = 27 runs)
#  2. pretrained ResNet-18: baseline vs oversampling (3 levels x 3 seeds = 18 runs)
$exp = Split-Path $PSScriptRoot -Parent
$py = 'D:\CudaEnvironmentForClaude\python.exe'
$env:PYTHONUNBUFFERED = '1'; $env:TORCH_HOME = 'D:/Datasets/torch_cache'
Add-Content -LiteralPath "$PSScriptRoot\queue.log" -Value ("`n=== {0}: step A - train on synthetic, test on real (27 runs) ===" -f (Get-Date -Format 'yyyy-MM-dd HH:mm'))
& $py -u "$exp\run_queue.py" --methods tstr_s1_cond_minority tstr_s2_cond_all tstr_s3_per_class --widths 1.0 --seeds 0 1 2
Add-Content -LiteralPath "$PSScriptRoot\queue.log" -Value ("`n=== {0}: step B - pretrained ResNet-18, baseline vs oversampling (18 runs) ===" -f (Get-Date -Format 'yyyy-MM-dd HH:mm'))
& $py -u "$exp\run_queue.py" --methods baseline oversampling --widths 1.0 --seeds 0 1 2 --arch resnet18
Add-Content -LiteralPath "$PSScriptRoot\queue.log" -Value ("=== CHECKS DONE {0} ===" -f (Get-Date -Format 'yyyy-MM-dd HH:mm'))
