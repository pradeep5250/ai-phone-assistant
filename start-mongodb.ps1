# Start portable MongoDB from D: (data also on D:)
$mongod = "D:\tools\mongodb\mongodb-win32-x86_64-windows-8.0.12\bin\mongod.exe"
$config = "D:\tools\mongodb\mongod.cfg"

if (-not (Test-Path $mongod)) {
    Write-Error "mongod not found at $mongod"
    exit 1
}

$running = Get-Process mongod -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "MongoDB already running (PID $($running.Id))"
    exit 0
}

New-Item -ItemType Directory -Force -Path "D:\data\mongodb","D:\data\mongodb\log" | Out-Null
Start-Process -FilePath $mongod -ArgumentList "--config", $config -WindowStyle Hidden
Write-Host "MongoDB started on 127.0.0.1:27017 (data: D:\data\mongodb)"
