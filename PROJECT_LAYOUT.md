# Project layout

- `src/`: maintained Python modules and `prepare_data.sh`; run commands from the repository root.
- `src/notebooks/ViSEF_complete.ipynb`: the active end-to-end workflow.
- `legacy/code/`: original tracked source snapshots from before the current updates. Use `src/` for new work.
- `legacy/notebooks/`: previous notebooks and their saved outputs.
- `legacy/results/`: prior plots and tables retained for reference. Old MIA and epsilon-sweep files are excluded from current conclusions.
- `results/`: destination for newly generated summary tables.
- `README.md`, `bao_cao_ViSEF.tex`, `bao_cao_ViSEF_preview.pdf`, and `template_bao_cao_khoa_hoc.tex` remain at the project root.
- `requirements.txt`: dependency list at the project root.

The MRI dataset and completed runs for seeds 123 and 456 are not included. The unified notebook runs three seeds and checks completeness before calculating uncertainty summaries.
