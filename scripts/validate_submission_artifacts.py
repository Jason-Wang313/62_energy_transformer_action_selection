from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER_PDF = ROOT / "paper" / "main.pdf"
DOWNLOADS_PDF = Path.home() / "Downloads" / "62.pdf"
DESKTOP_PDF = Path.home() / "Desktop" / "62.pdf"


def row_count(name: str) -> int:
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def pdf_pages(path: Path) -> int:
    proc = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not read page count from {path}")


def check(name: str, ok: bool, detail: object) -> bool:
    print(("PASS" if ok else "FAIL") + f"\t{name}\t{detail}")
    return ok


def main() -> int:
    ok = True
    ok &= check("main_rows", row_count("energy_action_raw.csv") == 11520, row_count("energy_action_raw.csv"))
    ok &= check("ablation_rows", row_count("energy_action_ablation_raw.csv") == 2112, row_count("energy_action_ablation_raw.csv"))
    ok &= check("metrics_rows", row_count("energy_action_metrics.csv") == 120, row_count("energy_action_metrics.csv"))
    ok &= check("ablation_metric_rows", row_count("energy_action_ablation.csv") == 22, row_count("energy_action_ablation.csv"))
    ok &= check("pairwise_rows", row_count("energy_action_pairwise.csv") == 99, row_count("energy_action_pairwise.csv"))
    ok &= check("paper_pdf_exists", PAPER_PDF.exists(), PAPER_PDF)
    ok &= check("downloads_pdf_exists", DOWNLOADS_PDF.exists(), DOWNLOADS_PDF)
    if PAPER_PDF.exists():
        ok &= check("paper_pdf_pages", pdf_pages(PAPER_PDF) >= 25, pdf_pages(PAPER_PDF))
    if DOWNLOADS_PDF.exists():
        ok &= check("downloads_pdf_pages", pdf_pages(DOWNLOADS_PDF) >= 25, pdf_pages(DOWNLOADS_PDF))
    ok &= check("desktop_pdf_absent", not DESKTOP_PDF.exists(), DESKTOP_PDF)
    if PAPER_PDF.exists() and DOWNLOADS_PDF.exists():
        ok &= check("downloads_matches_paper_size", PAPER_PDF.stat().st_size == DOWNLOADS_PDF.stat().st_size, f"{PAPER_PDF.stat().st_size}/{DOWNLOADS_PDF.stat().st_size}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

