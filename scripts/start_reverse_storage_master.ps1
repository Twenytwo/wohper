param(
    [string]$RepoRoot = "C:\SITI WEB BESA E PROGETTI\Progetti\H farm 1 anno\VPS Hermes, Wohper",
    [string]$Python = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [int]$FilePort = 9180,
    [int]$WorkerPort = 9200,
    [int]$LocalRelayPort = 9101,
    [string]$AllowedRemoteCidr = "172.20.10.0/28",
    [switch]$ConfigureFirewall,
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"

function Get-PortPids {
    param([int[]]$Ports)

    $found = @()
    foreach ($port in $Ports) {
        $found += netstat -ano |
            Select-String ":$port\s" |
            ForEach-Object {
                $parts = ($_ -replace '^\s+', '') -split '\s+'
                if ($parts.Count -ge 5) { [int]$parts[-1] }
            }
    }
    @($found | Where-Object { $_ -gt 0 } | Sort-Object -Unique)
}

function Get-PreferredIpv4 {
    $candidates = @()
    $interfaces = [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces()
    foreach ($interface in $interfaces) {
        if ($interface.OperationalStatus -ne [System.Net.NetworkInformation.OperationalStatus]::Up) {
            continue
        }
        $props = $interface.GetIPProperties()
        foreach ($addr in $props.UnicastAddresses) {
            if ($addr.Address.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
                continue
            }
            $ip = $addr.Address.ToString()
            if ($ip -eq "127.0.0.1" -or $ip.StartsWith("169.254.")) {
                continue
            }
            $score = 10
            if ($ip.StartsWith("172.20.10.")) { $score = 100 }
            elseif ($ip.StartsWith("192.168.")) { $score = 80 }
            elseif ($ip -match "^10\.") { $score = 70 }
            elseif ($ip -match "^172\.(1[6-9]|2[0-9]|3[0-1])\.") { $score = 60 }
            $candidates += [pscustomobject]@{
                IP = $ip
                Interface = $interface.Name
                Score = $score
            }
        }
    }

    if ($candidates.Count -eq 0) {
        throw "No usable IPv4 address found. Connect the master to the hotspot/Wi-Fi first."
    }
    ($candidates | Sort-Object Score -Descending | Select-Object -First 1).IP
}

function Wait-TcpPort {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutSeconds = 8
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

Set-Location -LiteralPath $RepoRoot
$logsDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$ports = @($FilePort, $WorkerPort, $LocalRelayPort)
if (-not $NoClean) {
    Write-Host "Cleaning old Wohper reverse-storage listeners..."
    foreach ($processId in (Get-PortPids -Ports $ports)) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
            Write-Host "  stopped pid $processId"
        } catch {
            Write-Host "  could not stop pid $processId"
        }
    }
}

$masterIp = Get-PreferredIpv4
Write-Host "Detected master IP: $masterIp"

if ($ConfigureFirewall) {
    Write-Host "Configuring firewall for hotspot range $AllowedRemoteCidr..."
    New-NetFirewallRule -DisplayName "Wohper Master File Server $FilePort Hotspot" `
        -Direction Inbound -Action Allow -Protocol TCP -LocalPort $FilePort `
        -RemoteAddress $AllowedRemoteCidr -Profile Any -ErrorAction SilentlyContinue | Out-Null
    New-NetFirewallRule -DisplayName "Wohper Reverse Worker $WorkerPort Hotspot" `
        -Direction Inbound -Action Allow -Protocol TCP -LocalPort $WorkerPort `
        -RemoteAddress $AllowedRemoteCidr -Profile Any -ErrorAction SilentlyContinue | Out-Null
}

$fileOut = Join-Path $logsDir "reverse_storage_file_server.out.log"
$fileErr = Join-Path $logsDir "reverse_storage_file_server.err.log"
$relayOut = Join-Path $logsDir "reverse_storage_master.out.log"
$relayErr = Join-Path $logsDir "reverse_storage_master.err.log"
Remove-Item -LiteralPath $fileOut, $fileErr, $relayOut, $relayErr -ErrorAction SilentlyContinue

Write-Host "Starting file server on 0.0.0.0:$FilePort"
Start-Process -FilePath $Python `
    -ArgumentList @("-u", "-m", "http.server", "$FilePort", "--bind", "0.0.0.0") `
    -WorkingDirectory (Join-Path $RepoRoot "tools") `
    -WindowStyle Hidden | Out-Null

Write-Host "Starting reverse relay: worker in :$WorkerPort -> local http 127.0.0.1:$LocalRelayPort"
Start-Process -FilePath $Python `
    -ArgumentList @(
        "-u",
        "tools\zc_reverse_storage_master.py",
        "--worker-host", "0.0.0.0",
        "--worker-port", "$WorkerPort",
        "--http-host", "127.0.0.1",
        "--http-port", "$LocalRelayPort"
    ) `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden | Out-Null

$fileReady = Wait-TcpPort -HostName "127.0.0.1" -Port $FilePort
$relayReady = Wait-TcpPort -HostName "127.0.0.1" -Port $LocalRelayPort

Write-Host ""
Write-Host "Listeners:"
netstat -ano | findstr ":$FilePort :$WorkerPort :$LocalRelayPort"

Write-Host ""
Write-Host "Health:"
Write-Host "  file server 127.0.0.1:$FilePort = $fileReady"
Write-Host "  worker ingress 0.0.0.0:$WorkerPort = listening (not probed; probing consumes the reverse-worker slot)"
Write-Host "  local relay 127.0.0.1:$LocalRelayPort = $relayReady"

$workerListening = (netstat -ano | Select-String ":$WorkerPort\s+.*LISTENING") -ne $null
if (-not ($fileReady -and $workerListening -and $relayReady)) {
    Write-Host ""
    Write-Host "One or more listeners failed. Logs:"
    Write-Host "  $fileOut"
    Write-Host "  $fileErr"
    Write-Host "  $relayOut"
    Write-Host "  $relayErr"
    exit 1
}

$workerBootstrapUrl = "http://$masterIp`:$FilePort/zc_worker_connect.ps1"
$workerCommand = "Invoke-WebRequest -UseBasicParsing $workerBootstrapUrl -OutFile C:\WohperStorage\zc_worker_connect.ps1; powershell -ExecutionPolicy Bypass -File C:\WohperStorage\zc_worker_connect.ps1 -MasterHost $masterIp"
$connectionCard = @"
Wohper reverse storage master

Master IP: $masterIp
File server: http://$masterIp`:$FilePort
Worker ingress: $masterIp`:$WorkerPort
Wohper endpoint: http://127.0.0.1:$LocalRelayPort

Worker one-liner:
$workerCommand

Master health test after worker connects:
powershell -ExecutionPolicy Bypass -File scripts\test_reverse_storage_master.ps1

Logs:
$fileOut
$fileErr
$relayOut
$relayErr
"@

$cardPath = Join-Path $logsDir "reverse_storage_connection.txt"
Set-Content -Encoding Ascii -LiteralPath $cardPath -Value $connectionCard

Write-Host ""
Write-Host "Master endpoint for Wohper:"
Write-Host "  http://127.0.0.1:$LocalRelayPort"
Write-Host ""
Write-Host "Worker one-liner:"
Write-Host "  $workerCommand"
Write-Host ""
Write-Host "Connection card saved:"
Write-Host "  $cardPath"
