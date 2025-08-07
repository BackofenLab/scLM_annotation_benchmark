## Single-cell and Language foundation models

### Workflow description:

#### download process
WIP: download.py script puts h5ad file into data (e.g. data/tabula_sapiens_blood.h5ad)

#### run_model process
Runs a run.sh bash script which is given the h5ad file, the model directory (should be models/${model_name}) and the model_name as well as dataset name.  
When creating a new model, implement a run.sh which installs the model and runs it on the given h5ad file, then create the file results/$dataset/$model/predictions.csv with the columns: "true_labels", "predicted_labels" from the output

#### evaluate_annotation process
Runs the evaluate.py script which takes as input the path of the predictions (i.e. results/$dataset/$model/predictions.csv), the name of the dataset and the model and returns results/$dataset/$model/evaluation.json  
TODO: It also creates a pdf with plots, currently this is not published but can only be found in the corresponding work directory.

### Planning excel sheet

[internal link removed]

### Meeting minutes

[internal link removed]
