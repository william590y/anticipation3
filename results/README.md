# Results (2026-08-27 session)

All artifacts are COPIES; the generators and canonical outputs live in
`figures/scalemax/`. Regeneration commands per artifact:

| file(s) | what | regenerate |
| --- | --- | --- |
| `table_aggregate.{png,pdf,tex}` | windowed test set, 4 metrics x 3 systems, window+piece level, sign-flip p | `python figures/scalemax/make_tables.py` |
| `table_perpiece.{png,pdf,tex}` | per-work breakdown, 14 works x 4 metrics x 3 systems | same script |
| `hist_{base,smax,ioi,ioi_smax}_windowed_test.png` | small-multiple F1 histograms (per window + per work) | `python figures/scalemax/make_histograms.py` |
| `hist_notelevel_onset_err_test.png` | per-note signed onset error, linear axis, +-25 clip | `python figures/scalemax/make_note_hist.py` |
| `hist_notelevel_onset_err_log_test.png` | same on a log axis (shows the derailed-regime tail) | inline variant of the above |

Data sources: `nbest_data/test_set_scalemax_f1.json`,
`nbest_data/test_set_selector_eval.json`, `visualizer/data_testset.js`.

Pending artifacts land here as their jobs finish: whole-song table (task 2/3),
val + rollout histograms (tasks 4/5), pitch-forcing table.

## LaTeX routing (`latex/` + `all_figures.pdf`)

`figures/scalemax/publish_results.py` (idempotent -- rerun after any stage):
copies every artifact here, writes one `latex/fig_<name>.tex` snippet per plot
(`\begin{figure}` + `\includegraphics` + caption, ready to `\input`), and
compiles `latex/all_figures.tex` -> `all_figures.pdf` (tectonic), which shows
every figure and table rendered in one LaTeX document. Pending artifacts
(whole-song table, rollout/val histograms, pitch-forcing table) are
pre-registered in the publisher and appear automatically once generated.
