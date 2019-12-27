# ADMET_RNN
ADMET property prediction using recursive neural network.

## Datasets

| Name | Description | size | task | location | source              |
|:-----|:------------|:-----|:-----|:---------|:--------------------|
| ESOL | solubility in water |1128|regression|datasets/ESOL.txt|J. S. Delaney, J. Chem. Inf. Model., 2004, 44, 1000–1005|
| HIV |anti-HIV activity|34092|classification|datasets/HIV.txt|https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data|

## Structure
### base.py
- Node: Node for Tree and Graph.
- Tree
- Graph
### model.py
- FeatureRNN: feature learning to get feature vector
- SolNet: NN to predict solubility
- PlusSolNet: features get by FeatureRNN are concat with global features calculated by rdkit
- HIVNet: NN to predict anti-hiv activity
- PlusHIVNet: features get by FeatureRNN are concat with global features calculated by rdkit
### train.py
- loss: function