# Data directory

This directory is intentionally empty in the repository.

The analysis uses the **IIP Patent Database** (Institute of Intellectual
Property, Tokyo), which cannot be redistributed under its licensing terms.

To reproduce the analysis, place the raw decade files here:

```
data/
├── ap_1990s.txt ... ap_2020s.txt          (application records)
├── applicant_1990s.txt ... applicant_2020s.txt   (applicant/co-applicant records)
├── cc_1990s.txt ... cc_2020s.txt          (citation/reason-code records)
└── hr_1990s.txt ... hr_2020s.txt          (examination history records)
```

Then run the pipeline in `src/01_data_construction/`, which will produce
the cached `.pkl` files (`df_application_v3_clean.pkl`, `panel_v3_clean.pkl`,
etc.) consumed by all downstream scripts. Update the `CACHE_DIR` variable
at the top of each script to point at wherever you keep these cached files
(this repository's scripts default to the current directory, `.`).
