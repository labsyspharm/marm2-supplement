#!/usr/bin/env bash
export N_JOBS=100
export N_THREADS=1
source ./venv/bin/activate
python -m snakemake process_all_data --cores 6