# Paper 62 VLA Highlight Hardening Plan

Objective: make Paper 62's boxed PDF highlights match the VLA-v4 role-model PDF while preserving the frozen RC-RET evidence and STRONG_REVISE scientific conclusion.

## Role-Model Target

- Citation links: green rectangular border, no fill, default tight hyperref box.
- Internal references: red rectangular border, no fill, default tight hyperref box.
- Border: `pdfborder={0 0 1}` to match the VLA-v4 annotation border width.
- Typography/layout: no font, spacing, table, caption, or figure content changes unless visual QA exposes a layout defect.

## Current Paper 62 Mismatch

- `Downloads/62.pdf` has orange citation boxes on pages 1, 9, and 10.
- `Downloads/62.pdf` has blue internal-reference boxes on page 13.
- The source cause is `paper/main.tex`, where `citebordercolor` is orange and `linkbordercolor` is blue.

## Execution Plan

1. Keep RAM use low by rendering only affected pages before and after the edit: pages 1, 9, 10, and 13.
2. Change only the `hyperref` border colors in `paper/main.tex`:
   - `citebordercolor={0 1 0}`
   - `linkbordercolor={1 0 0}`
   - `urlbordercolor={0 1 0}`
   - preserve `pdfborder={0 0 1}`
3. Rebuild with the existing repository build script.
4. Validate the rebuilt PDF annotation metadata with `pypdf`.
5. Render affected pages again and visually compare with the VLA-v4 role model.
6. Copy/export only the accepted final PDF to `C:\Users\wangz\Downloads\62.pdf`.
7. Update status/build metadata if needed, remove temporary renders after QA, then commit and push a clean repo.

## Non-Goals

- Do not rerun MuJoCo/PyTorch experiments.
- Do not change the final STRONG_REVISE decision or scientific results.
- Do not pad pages or alter paper scope for a purely visual link-box fix.
