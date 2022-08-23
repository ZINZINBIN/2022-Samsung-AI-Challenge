import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.utils import extract_alphabet_dict
from src.CustomDataset import SMILESDataset
from src.loss import RMSELoss
from src.self_attention import SelfAttention
from src.train import train
from src.evaluate import evaluate
from src.inference import inference

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:0" 
else:
    device = 'cpu'

# cache init
torch.cuda.empty_cache()

if __name__ == "__main__":
    PATH = "./dataset/"

    train_df = pd.read_csv(os.path.join(PATH, "train_set.csv"))
    test_df = pd.read_csv(os.path.join(PATH, "test_set.csv"))

    total_smiles = pd.concat([train_df[['SMILES','index']] , test_df[['SMILES', 'index']]], axis = 0).reset_index(drop = True)
    char2idx,idx2char = extract_alphabet_dict(total_smiles)
    
    df_train, df_test = train_test_split(train_df, test_size = 0.2, random_state = 42)
    df_train, df_valid = train_test_split(df_train, test_size = 0.2, random_state = 42)

    train_data = SMILESDataset(char2idx, df_train, mode = 'train')
    valid_data = SMILESDataset(char2idx, df_valid, mode = 'train')
    test_data = SMILESDataset(char2idx, df_test, mode = 'train')

    batch_size = 128
    num_epoch = 16
    verbose = 4
    vocab_size = len(char2idx.items())
    dropout = 0.25
    embedd_config = {
        'embedd_dim' : 12,
        'hidden_dim_l0' : 32,
        'hidden_dim_l1' : 32,
        'hidden_dim_l2' : 64
    }

    train_loader = DataLoader(train_data, batch_size, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size, shuffle = True)
    
    model = SelfAttention(batch_size, vocab_size, dropout, embedd_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    loss_fn = RMSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)
  
    train_loss, valid_loss = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        num_epoch = num_epoch,
        verbose = verbose,
        root_dir = "./weights",
        best_pt = "self_attention_best.pt",
        last_pt = "self_attention_last.pt",
        max_norm_grad = 1.0
    )
