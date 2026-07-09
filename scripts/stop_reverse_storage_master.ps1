param(
    [int[]]$Ports = @(9180, 9200, 9101)
)

$ErrorActionPreference = "Stop"

$pids = @()
foreach ($port in $Ports) {
    $pids += Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        Where-Object { $_.OwningProcess -gt 0 } |
        Select-Object -ExpandProperty OwningProcess
    $pids += netstat -ano |
        Select-String ":$port\s" |
        ForEach-Object {
            $parts = ($_ -replace '^\s+', '') -split '\s+'
            if ($parts.Count -ge 5) { [int]$parts[-1] }
        }
}

$unique = @($pids | Sort-Object -Unique)
if ($unique.Count -eq 0) {
    Write-Host "No Wohper reverse-storage listeners found."
    exit 0
}

foreach ($processId in $unique) {
    try {
        Stop-Process -Id $processId -Force -ErrorAction Stop
        Write-Host "Stopped pid $processId"
    } catch {
        Write-Host "Could not stop pid $processId"
    }
}
