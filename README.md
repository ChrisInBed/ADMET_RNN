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
- PlusSolNet: 196 global features calculated by rdkit are concat with the recursive features acquired by FeatureRNN
- HIVNet: NN to predict anti-hiv activity
- PlusHIVNet: 196 global features calculated by rdkit are concat with the recursive features acquired by FeatureRNN
### load_data.py
present iterable data loader with defined batch_size, to feed a model.
A data loader for each explicit Model in model.py
### train.py
- train: base train function, no need to explain
- ModelCV: Cross-Validation wrapper of model in model.py. ModelCV.train method will preform k-fold
  cross validation automatically
### evaluate.py
Provide an example to evaluate trained models on test data set

## Evaluation
- SolNet和PlusSolNet完成回归任务，使用RMSE评估就可以。
- HIVNet和PlusHIVNet完成分类任务，但数据集极不平衡(有活性:无活性大约为1/40)，所以训练时注意调整损失函数的权重
  模型评估同时采用精度和回收率。
- 调参(神经元数量、激活函数和外层网络深度)，可进行手动调参(对于深度网络工作量较大，所以凭感觉调一调就好)，网格调参
  和随机调参可获得更科学的结果，[hyperopt](https://github.com/hyperopt/hyperopt)模块提供了调参接口，但最高效的
  是贝叶斯调参，该模块尚未完成这一部分
- 尽量使用图形描述模型性能，如对于回归任务，绘制实验值-预测值散点图，求回归直线R2，可绘制指标-epoch折线图描述训练进程，
  另外，记录训练时间也有助于反映训练效率

## Report
- 背景，请参考文献工作
- 模型原理，参考报告内容，papers/report_191213.docx
- 数据集介绍，数据集预处理与符号说明，以及模型参数
- 模型评估，并与参考文献成果对比
- 改进建议
