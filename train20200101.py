import torch
from torch import nn

from model import SolNet, PlusSolNet, HIVNet, PlusHIVNet
from load_data import SolData, PlusSolData, HIVData, PlusHIVData
from train import ModelCV

model_cv = ModelCV(SolNet, folds=5, inner_hidden_size=150, feature_size=50, hidden_size=100)
model_cv.load_model('solnet_model')
data_loader = SolData('datasets/esol_train.txt', batch_size=64)
Optimizer = torch.optim.Adam
optim_dict = {'lr': 1e-3}
loss = nn.MSELoss()
model_cv.train(data_loader, Optimizer, optim_dict, loss, early_stopping=True, tol=0.5)
model_cv.save_model('solnet_model')
# model_cv.predict(["CCCO", "c1ccc(N)cc1C=O"])
