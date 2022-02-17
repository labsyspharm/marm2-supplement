# Supplementary Material for the MARM2.0 Model

## Setup

To set up MARM2.0, run `install_as_package.sh`. This will install all dependencies and
MARM as in-source python package. If installation of independencies fails, please see the 
documentation of the respective package (e.g., 
[AMICI documentation](https://amici.readthedocs.io/en/latest/python_installation.html))

As workflow manager for model calibration and analysis, MARM2.0 uses Snakemake, which 
requires cluster configuration through a JSON file. This JSON file must be 
named `cluster.json`, 
please see the 
[Snakemake documentation](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html) 
for more details on how to create such a file.

The shell scripts in this repository are designed to work with a SLURM 
Workload  Manager and  use the `sbatch` command to submit cluster jobs. On 
different systems, this may need to be  adapted by editing the shell scripts.

## Customization

All steps described below can be customized by editing the snakemake configuration file 
`Snakefile`. 
* `MODEL` controls which model will be analysed. Models need to be stored in the 
`MARM/pysb_flat` directory, follow the `$MODEL_$VARIANT.py` syntax and populate a PySB 
model in the `model` workspace variable.
* `VARIANTS` variable controls which model variant will analysed. Variants can be specified 
using the `$MODEL_$VARIANT.py` syntax (see above).
* `DATASETS` controls which perturbation data will be used. Available perturbations are `EGF` 
(EGF stimulation data), `EGFR` EGFR up/downregulation using CRISPR, `MEKi` MEK inhibitor 
dose response data, `PRAFi` panRAFi inhibitor dose reponse data and `RAFi` RAF 
inihibitor dose response data. When multiple perturbations
are to be used, tokens need to be joined by `_` and sorted alphabetically 
(eg., `EGF_EGFR_MEKi_PRAFi_RAFi`).

## Model Formulation

Documentation of model formulation is provided as interactive jupyter notebook 
`Model Documentation.ipynb`.

## Model Statistics 

Various statistics regarding model and data size are provided in the 
jupyter notebook `Model Statistics.ipynb`. This notebook was used to 
compute numbers listed throughout the text.

Channel specific analysis was performed using the `Stastistics Channels.ipynb` 
jupyter notebook.

## Calibration

Model calibration can be run by executing the `runEstimationCluster.sh` shell script. This 
script sets the number of optimization runs through the `N_JOBS` environment variable and the 
number of local starts per optimization run throught the `MS_PER_JOB` environment variable. 
Each optimization run is embarrassingly parellelized as individual cluster job. Within the scope 
of a single job, the `N_THREADS` environment variable controls the number of parallel local 
optimizations.

Calibration will populate results in the `MARM/results` folder. To aggregate results and 
update the parameters that are used in downstream steps, the user has to run the 
`collectResultsEstimation.py` python script, which requires `MODEL`, `VARIANT` and 
`DATASET` as arguments (default: `RTKERK base EGF_EGFR_MEKi_PRAFi_RAFi`).

Calibration requires up to 13-14 years of CPU time and should be executed on a cluster.

## Multi-Model Benchmark

The multi-model and full-model benchmark can be run by executing the  
`runBenchmarkCluster.sh` shell script. Benchmarking will populate results in the 
`MARM/analysis` folder. To aggregate results the, user has to run the 
`collectResultsBenchmark.py` python script, which requires `MODEL`, `VARIANT` and 
`DATASET` as arguments (default: `RTKERK base EGF_EGFR_MEKi_PRAFi_RAFi`)

Benchmarking may require up to 4 years of CPU time and should be executed on a 
cluster. Results were analysed and visualized using the `Speedup.ipynb` 
jupyter notebook.

## Analysis

Model analysis can be run by executing the `runAnalysisCluster.sh` shell script. 
Model analysis will employ parameter values stored in `MARM/parameters` that were 
generated by executing the `collectResultsEstimation.py` python script. For convenience,
this repository contains prepopulated estimation results that were used for the generation of 
figures in the manuscript. Analysis will populate results in the `MARM/analysis` folder.

Analysis may require up to 100 days of CPU time and should be executed on a cluster.

## Visualization

Figures shown in the manuscript can be generated by executing the 
`generateFiguresCluster.sh` shell script. Figure generation requires analysis results 
generated by `runAnalysisCluster.sh`. Generated graphs are stored in 
`MARM/figures`. This executes the following python scripts:

* `plot_comboprediction.py` generates graphs for Figures 5 and 6C,D,E,F
* `plot_feedback.py` generates graphs for Figures 3A and 6A
* `plot_finepulse.py` generates graphs for Figure 2A.
* `plot_mutRASprediction.py` generates graphs for Figure 8B,C,D
* `plot_panrafcomboprediction.py` generates graphs for Figure 7B,C and S7
* `plot_singleprediction.py` generates graphs for Figure 7A
* `plot_trainingdata.py` generates graphs for Figures 2E,F, 6A,B, S2, S3 and S6
* `plot_transduction.py` generates graphs for Figures 4B,C,D,E,F and S4

Estimation results can be visualized using the `plot_fits.py` python script,
which was used to generate graphs for Figure S1A,B.

Figures 1B,C were generated using the `Model Statistics.ipynb` notebook.
Figures 1D,E were generated using the `Speedup.ipynb` notebook.
Figure 2D was generated using the `Stastistics Channels.ipynb` notebook.


