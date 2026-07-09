param(
    [int]$LocalRelayPort = 9101,
    [string]$ExpertName = "layer12_expert45.zcblk"
)

$ErrorActionPreference = "Stop"

function Show-Step {
    param([string]$Text)
    Write-Host ""
    Write-Host "== $Text =="
}

Show-Step "Relay health"
$health = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$LocalRelayPort/health" -TimeoutSec 10
Write-Host $health.Content

Show-Step "Optional expert fetch"
$expertUrl = "http://127.0.0.1:$LocalRelayPort/experts/$ExpertName"
try {
    $response = Invoke-WebRequest -UseBasicParsing $expertUrl -TimeoutSec 10
    $bytes = [byte[]]$response.Content
    Write-Host "Fetched $($bytes.Length) bytes from $expertUrl"
    if ($bytes.Length -le 4096) {
        Write-Host ([System.Text.Encoding]::ASCII.GetString($bytes))
    }
} catch {
    Write-Host "Expert fetch did not pass: $($_.Exception.Message)"
    Write-Host "This is OK if no dummy/real shard exists yet."
}

Show-Step "Connections"
netstat -ano | findstr ":9180 :9200 :9101"
