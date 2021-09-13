#!/usr/bin/env bash
export N_JOBS=100
export N_THREADS=1
snakemake generate_figures -j 100 --cluster-config cluster.json --cluster "sbatch -p {cluster.queue} -n 1 -c {cluster.cores} -N 1 -t {cluster.time} --mem={cluster.memory} -e {cluster.error} -o {cluster.output} --mail-user={cluster.mail}"