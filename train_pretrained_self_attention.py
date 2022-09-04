import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.utils import extract_alphabet_dict, plot_training_curve
from src.CustomDataset import SMILESDataset
from src.loss import RMSELoss
from src.self_attention import Regressor, SelfAttention, SelfAttentionEncoder, SelfAttentionNetwork
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

    # qm9 dataset
    qm9 = pd.read_csv(os.path.join(PATH, "qm9.csv"))
    qm9.rename(columns = {'smiles':'SMILES', 'mol_id':'index'}, inplace = True)

    # alphabet extraction
    char2idx, idx2char = extract_alphabet_dict(qm9[['SMILES','index']])

    df_train, df_test = train_test_split(train_df, test_size = 0.2, random_state = 42)
    df_train, df_valid = train_test_split(df_train, test_size = 0.2, random_state = 42)
    
    cols = ["Reorg_g", "Reorg_ex"]

    train_data = SMILESDataset(char2idx, df_train, mode = 'train', cols = cols)
    valid_data = SMILESDataset(char2idx, df_valid, mode = 'train', cols = cols)
    test_data = SMILESDataset(char2idx, df_test, mode = 'train', cols = cols)

    batch_size = 128
    num_epoch = 64
    verbose = 4
    vocab_size = len(char2idx.items()) + 1
    dropout = 0.25
    embedd_config = {
        'embedd_dim' : 12,
        'hidden_dim_l0' : 64,
        'hidden_dim_l1' : 64,
        'hidden_dim_l2' : 128
    }

    train_loader = DataLoader(train_data, batch_size, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size, shuffle = True)

    encoder = SelfAttentionEncoder(batch_size, vocab_size, dropout, embedd_config).to(device)
    regressor = Regressor(encoder.linear_input_dims, len(cols)).to(device)

    # load pretrained-weight
    encoder.load_state_dict(torch.load("./weights/self_attention_encoder_best.pt"), strict = False)

    model = SelfAttentionNetwork(encoder, regressor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-4)
    
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
        max_norm_grad = 1.0,
        mode = 'BERT'
    )

    plot_training_curve("./result/self_attention_learning_curve.png", train_loss, valid_loss)
    model.load_state_dict(torch.load("./weights/self_attention_best.pt"), strict = False)
    test_loss = evaluate(test_loader, model, optimizer, loss_fn, device, 'BERT')
