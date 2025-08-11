## Single-cell and Language foundation models

### Workflow description:

#### nextflow.config
Config file, currently only has three relevant parameters:  
`datasets`: list of datasets to use. Every dataset needs a yaml file with the same name (`dataset.yaml`) in the `datasets` directory  
`models`: list of models to use. Needs a `run.sh` in `models/${model_name}/src/run.sh`  
`data_cache`: a directory to save/cache the downloaded data and the created subsets. Within the work directory, only links to this data are stored, to avoid storing data multiple times.  

There is a profile `local` for running locally and `helix` for running on the helix cluster. TODO: Other/more general slurm clusters

#### Run main.nf
To run `main.nf` type: `nextflow run main.nf -profile local -with-conda`
This defines the profile to use (local, in this example) and enables Conda support for the run globally. 

#### download process
WIP: download.py script puts h5ad file into data (e.g. data/tabula_sapiens_blood.h5ad)

#### run_model process
Runs a run.sh bash script which is given five inputs: the h5ad file, the model directory (should be models/${model_name}), the name of the dataset, the dataset config file, the name of the model.
**The run.sh should take three inputs**: 1. The h5ad file 2. the output path of the predictions.csv 3. The dataset config
When creating a new model, implement a run.sh which installs the model and runs it on the given h5ad file, then create the file results/$dataset/$model/predictions.csv with the columns: "true_labels", "predicted_labels" from the output

#### evaluate_annotation process
Runs the evaluate.py script which takes as input the path of the predictions (i.e. results/$dataset/$model/predictions.csv), the name of the dataset and the model and returns results/$dataset/$model/evaluation.json  
TODO: It also creates a pdf with plots, currently this is not published but can only be found in the corresponding work directory.

### Planning excel sheet

[internal link removed]

### Meeting minutes

[internal link removed]
