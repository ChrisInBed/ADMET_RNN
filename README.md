# ADMET_RNN
ADMET property prediction using recursive neural network.
We developed 4 models to predict molecule's solubility in water and its anti-HIV activity

## Datasets

| Name | Description | size | task | location | source              |
|:-----|:------------|:-----|:-----|:---------|:--------------------|
| ESOL | solubility in water |1128|regression|datasets/ESOL.txt|J. S. Delaney, J. Chem. Inf. Model., 2004, 44, 1000–1005|
| HIV |anti-HIV activity|34092|classification|datasets/HIV.txt|https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data|

smiles expressions that can't be recognized by rdkit were droped.(A small population compared to the bulk dataset)
## Structure
### base.py
- Node: Node for Tree and Graph.
- Tree
- Graph
### model.py
all models are inherited from nn.Module
- FeatureRNN: feature learning to get feature vector
- SolNet: NN to predict solubility
- PlusSolNet: 196 features get by FeatureRNN are concat with global features calculated by rdkit
- HIVNet: NN to predict anti-hiv activity
- PlusHIVNet: 196 features get by FeatureRNN are concat with global features calculated by rdkit
### load_data.py
present iterable data loader with defined batch_size, to feed a model.
A data loader for each explicit Model in model.py
### train.py
- train: base train function, no need to explain
- ModelCV: Cross-Validation wrapper of model in model.py. ModelCV.train method will preform k-fold
  cross validation automatically
