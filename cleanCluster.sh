#!/usr/bin/env bash
export N_JOBS=100
export MS_PER_JOB=1
export N_THREADS=1
snakemake clean
snakemake --unlock
