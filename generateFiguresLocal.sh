#!/usr/bin/env bash
export N_JOBS=100
export N_THREADS=1
source ./venv/bin/activate
python -m snakemake generate_figures --cores 4