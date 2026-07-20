param(
    [string]$DestinationRoot = "$PSScriptRoot/../data/aishell1",
    [int]$Segments = 2
)

$ErrorActionPreference = "Stop"
$dataUrl = "https://openslr.elda.org/resources/33/data_aishell.tgz"
$resourceUrl = "https://openslr.elda.org/resources/33/resource_aishell.tgz"
$dataSize = 15582913665L
$resourceSize = 1246920L
$downloadDir = Join-Path $DestinationRoot "downloads"
$dataArchive = Join-Path $downloadDir "data_aishell.tgz"
$resourceArchive = Join-Path $downloadDir "resource_aishell.tgz"
$partsDir = Join-Path $downloadDir "data_aishell.parts"

New-Item -ItemType Directory -Force -Path $downloadDir, $partsDir | Out-Null

if (-not (Test-Path $dataArchive) -or (Get-Item $dataArchive).Length -ne $dataSize) {
    $segmentSize = [math]::Ceiling($dataSize / $Segments)
    $processes = @()
    for ($index = 0; $index -lt $Segments; $index++) {
        $start = [int64]($index * $segmentSize)
        $end = [int64][math]::Min($dataSize - 1, $start + $segmentSize - 1)
        $expectedLength = $end - $start + 1
        $part = Join-Path $partsDir ("part-{0:D2}" -f $index)
        if ((Test-Path $part) -and (Get-Item $part).Length -eq $expectedLength) {
            continue
        }
        $quotedPart = '"' + $part + '"'
        $arguments = @(
            "-k", "--ssl-no-revoke", "--http1.1", "-L", "--fail",
            "--retry", "12", "--retry-delay", "5",
            "--range", "$start-$end", "-o", $quotedPart, $dataUrl
        )
        $processes += [pscustomobject]@{
            Process = Start-Process -FilePath "curl.exe" -ArgumentList $arguments -PassThru -WindowStyle Hidden
            Part = $part
            ExpectedLength = $expectedLength
        }
    }

    foreach ($entry in $processes) {
        $entry.Process.WaitForExit()
        if ($entry.Process.ExitCode -ne 0) {
            throw "AISHELL segment download failed: $($entry.Part)"
        }
        if ((Get-Item $entry.Part).Length -ne $entry.ExpectedLength) {
            throw "AISHELL segment has an unexpected size: $($entry.Part)"
        }
    }

    $output = [System.IO.File]::Create($dataArchive)
    try {
        for ($index = 0; $index -lt $Segments; $index++) {
            $part = Join-Path $partsDir ("part-{0:D2}" -f $index)
            $input = [System.IO.File]::OpenRead($part)
            try {
                $input.CopyTo($output, 8MB)
            } finally {
                $input.Dispose()
            }
        }
    } finally {
        $output.Dispose()
    }
}

if ((Get-Item $dataArchive).Length -ne $dataSize) {
    throw "Merged AISHELL archive has an unexpected size."
}

if (-not (Test-Path $resourceArchive) -or (Get-Item $resourceArchive).Length -ne $resourceSize) {
    & curl.exe -k --ssl-no-revoke --http1.1 -L --fail --retry 12 `
        --retry-delay 5 -o $resourceArchive $resourceUrl
    if ($LASTEXITCODE -ne 0) {
        throw "AISHELL resource download failed."
    }
}

if ((Get-Item $resourceArchive).Length -ne $resourceSize) {
    throw "AISHELL resource archive has an unexpected size."
}

Write-Output "AISHELL-1 archives downloaded and size-verified."
