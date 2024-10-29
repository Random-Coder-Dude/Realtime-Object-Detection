# Find all empty directories and add a .gitkeep file to each
Get-ChildItem -Recurse -Directory | Where-Object { $_.GetFiles().Count -eq 0 -and $_.GetDirectories().Count -eq 0 } | ForEach-Object {
    New-Item -Path $_.FullName -Name ".gitkeep" -ItemType File -Force
}

Write-Output ".gitkeep files added to all empty folders."
