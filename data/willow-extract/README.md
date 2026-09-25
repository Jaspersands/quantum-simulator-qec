# Willow extract

The first 2,000 shots of three of Google Quantum AI's Willow surface-code memory experiments
(Z basis, 30 rounds; patches `d3_at_q6_7`, `d5_at_q4_7` and `d7_at_q6_7`), for the live panel in
section 11 of the site. The files are Google's, unmodified except that the text files (circuits
and error model) are gzipped and every binary file is cut to 2,000 shots.

Source: Google Quantum AI, data for "Quantum error correction below the surface code threshold",
Zenodo record [13273331](https://zenodo.org/records/13273331), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `manifest.json` records each file's
origin and the SHA-256 of Google's detection events for these shots.

Regenerate with `.venv/bin/python tools/google.py extract`.
