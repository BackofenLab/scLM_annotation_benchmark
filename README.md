## Single-cell and Language foundation models

### Workflow description

#### nextflow.config

Config file, currently only has three relevant parameters:  
`datasets`: list of datasets to use. Every dataset needs a yaml file with the same name (`dataset.yaml`) in the `datasets` directory  
`models`: list of models to use. Needs a `run.sh` in `models/${model_name}/src/run.sh`  
`cache_dir`: a directory to save/cache the downloaded data and the created subsets. Within the work directory, only links to this data are stored, to avoid storing data multiple times. This parameter is also passed to the model and evaluate processes, since they often also download large files (eg. model weights, cellontology graph).

There is a profile `local` for running locally and `helix` for running on the helix cluster. TODO: Other/more general slurm clusters

#### Run main.nf

To run `main.nf` type: `nextflow run main.nf -profile local`
This defines the profile to use (local, in this example),
Or alternatively: `nextflow run main.nf -with-conda`
to enables Conda support for the run globally.

#### download process

download.py script puts h5ad file into cache/download (e.g. cache/download/tabula_sapiens_blood.h5ad). Subsets are also stored as additional files.

#### run_model process

**For every model**: Add a Conda environment file named `<model>.yml` to `envs/models/`. The model’s `run.sh` script will create and use the Conda environment defined in that file.
Runs a run.sh bash script which is given five inputs: the h5ad file, the model directory (should be models/${model_name}), the name of the dataset, the dataset config file, the name of the model.

#### prepare_model process

If a model needs to clone git repositories etc. It is recommended to do that in a prepare.sh file, such that these repositories can be pip installed through the `<model>.yml` in `envs/models/`. Installing repositories from within the run.sh can lead to complications with nextflows parallelization.

**The run.sh should take three(four) inputs**: 1. The h5ad file 2. the output path of the predictions.csv 3. The dataset config (4. The cache_dir. Implement if needed)
When creating a new model, implement a run.sh which installs the model and runs it on the given h5ad file, then create the file results/$dataset/$model/predictions.csv with the columns: "true_labels", "predicted_labels" from the output

**For LLMs (e.g. DeepSeekV3.2 or GPT-5.4)**: You must provide your own [OpenRouter](https://openrouter.ai/) API key. There are two recommended ways to do this:

1. **Environment variable (recommended):** Set `OPENROUTER_API_KEY` in your shell before running Nextflow:

   ```bash
   export OPENROUTER_API_KEY=sk-or-v1-...
   nextflow run main.nf -profile local
   ```

   `nextflow.config` reads this variable at startup via `System.getenv()` and forwards it into all process script blocks automatically. This works reliably across all executors (local and SLURM). To avoid typing the export every time, add it to your shell profile (`~/.bashrc` or `~/.zshrc`).

2. **Nextflow command-line flag:** Pass the key directly as a param override:

   ```bash
   nextflow run main.nf --openrouter_api_key sk-or-v1-... -profile local
   ```

**Never commit or push your API key to the GitHub repository.**

#### evaluate_annotation process

Runs the evaluate.py script which takes as input the path of the predictions (i.e. results/$dataset/$model/predictions.csv), the name of the dataset and the model and returns results/$dataset/$model/evaluation.json  
TODO: It also creates a pdf with plots, currently this is not published but can only be found in the corresponding work directory.

### Planning excel sheet

<[internal link removed]>

### Meeting minutes

<[internal link removed]>
