$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

python scripts\render_latex_tables.py
python scripts\summarize_final_decision.py

Set-Location (Join-Path $Root "paper")

pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

$DownloadsPdf = Join-Path $HOME "Downloads\62.pdf"
Copy-Item -LiteralPath "main.pdf" -Destination $DownloadsPdf -Force

$DesktopPdf = Join-Path $HOME "Desktop\62.pdf"
if (Test-Path $DesktopPdf) {
    throw "Desktop PDF exists unexpectedly: $DesktopPdf"
}

Write-Output "Wrote $DownloadsPdf"

