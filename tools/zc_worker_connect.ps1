param(
    [Parameter(Mandatory = $true)]
    [string]$MasterHost,
    [int]$MasterFilePort = 9180,
    [int]$MasterWorkerPort = 9200,
    [string]$StorageRoot = "C:\WohperStorage",
    [int]$StoragePort = 9100,
    [string]$PythonLauncher = "py",
    [switch]$CleanStorageListener
)

$ErrorActionPreference = "Stop"

function Wait-TcpPort {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutSeconds = 10
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $iar = $client.BeginConnect($HostName, $Port, $null, $null)
            if ($iar.AsyncWaitHandle.WaitOne(500, $false)) {
                $client.EndConnect($iar)
                return $true
            }
        } catch {
        } finally {
            $client.Close()
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Stop-PortListener {
    param([int]$Port)

    $pids = netstat -ano |
        Select-String ":$Port\s" |
        ForEach-Object {
            $parts = ($_ -replace '^\s+', '') -split '\s+'
            if ($parts.Count -ge 5 -and $parts[3] -eq "LISTENING") { [int]$parts[-1] }
        } |
        Sort-Object -Unique

    foreach ($processId in $pids) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
            Write-Host "Stopped old storage listener pid $processId"
        } catch {
            Write-Host "Could not stop storage listener pid $processId"
        }
    }
}

if (-not (Get-Command $PythonLauncher -ErrorAction SilentlyContinue)) {
    throw "Python launcher '$PythonLauncher' not found. Install Python 3 or pass -PythonLauncher."
}

New-Item -ItemType Directory -Force -Path $StorageRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $StorageRoot "experts") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $StorageRoot "logs") | Out-Null

$baseUrl = "http://$MasterHost`:$MasterFilePort"
$storageScript = Join-Path $StorageRoot "zc_expert_storage_server.py"
$reverseScript = Join-Path $StorageRoot "zc_reverse_storage_worker.py"

Write-Host "Wohper worker bootstrap"
Write-Host "  master: $MasterHost"
Write-Host "  storage root: $StorageRoot"
Write-Host ""

Write-Host "Testing master file server $baseUrl..."
if (-not (Wait-TcpPort -HostName $MasterHost -Port $MasterFilePort -TimeoutSeconds 8)) {
    throw "Cannot reach master file server at $MasterHost`:$MasterFilePort. Check hotspot and master firewall."
}

Invoke-WebRequest -UseBasicParsing "$baseUrl/zc_expert_storage_server.py" -OutFile $storageScript
Invoke-WebRequest -UseBasicParsing "$baseUrl/zc_reverse_storage_worker.py" -OutFile $reverseScript
Write-Host "Downloaded worker scripts."

if ($CleanStorageListener) {
    Stop-PortListener -Port $StoragePort
}

$storageOut = Join-Path $StorageRoot "logs\expert_storage_server.out.log"
$storageErr = Join-Path $StorageRoot "logs\expert_storage_server.err.log"
Remove-Item -LiteralPath $storageOut, $storageErr -ErrorAction SilentlyContinue

if (-not (Wait-TcpPort -HostName "127.0.0.1" -Port $StoragePort -TimeoutSeconds 1)) {
    Write-Host "Starting local expert storage on 127.0.0.1:$StoragePort..."
    Start-Process -FilePath $PythonLauncher `
        -ArgumentList @("-u", $storageScript, "--root", $StorageRoot, "--host", "0.0.0.0", "--port", "$StoragePort") `
        -WindowStyle Hidden `
        -RedirectStandardOutput $storageOut `
        -RedirectStandardError $storageErr | Out-Null
}

if (-not (Wait-TcpPort -HostName "127.0.0.1" -Port $StoragePort -TimeoutSeconds 8)) {
    throw "Local expert storage did not start. Check $storageOut and $storageErr."
}

$health = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$StoragePort/health" -TimeoutSec 8
Write-Host "Local storage health: $($health.StatusCode)"
Write-Host ""
Write-Host "Connecting reverse worker to $MasterHost`:$MasterWorkerPort..."
Write-Host "Keep this window open. Wohper requests will appear here."
Write-Host ""

& $PythonLauncher -u $reverseScript --master-host $MasterHost --master-port $MasterWorkerPort --storage-base "http://127.0.0.1:$StoragePort" --storage-root $StorageRoot
