#!/usr/bin/env bash
snakemake run_analysis -j 10000 --cluster-config cluster.json --cluster "sbatch -p {cluster.queue} -n 1 -c {cluster.cores} -N 1 -t {cluster.time} --mem={cluster.memory} -e {cluster.error} -o {cluster.output} --mail-user={cluster.mail}" &
