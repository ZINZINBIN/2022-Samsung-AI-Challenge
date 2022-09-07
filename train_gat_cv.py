from sklearn.model_selection import KFold
import time
from typing import Optional, List, Literal, Union
import os
import torch
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from src.preprocessing import generate_dataloader_cv, generate_dataloader
from src.loss import RMSELoss
from src.model import GATNet
from src.train import train_per_epoch, valid_per_epoch
from src.evaluate import evaluate
from src.inference import inference
import gc

def train_cv(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    num_epoch : int = 64,
    root_dir : str = "./weights",
    best_pt : str = "best.pt",
    last_pt : str = "last.pt",
    max_norm_grad : Optional[float] = None,
    early_stopping_round : int = 4,
    ):

    train_loss_list = []
    valid_loss_list = []

    best_epoch = 0
    best_loss = torch.inf

    esr = 0
    is_early_stopping = False

    for epoch in range(num_epoch):
        train_loss = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad
        )
        valid_loss = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device 
        )
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        # save the lastest model
        torch.save(model.state_dict(), os.path.join(root_dir, last_pt))

        # Early Stopping
        if best_loss > valid_loss:
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), os.path.join(root_dir, best_pt))
            esr = 0
        elif best_loss < valid_loss:
            esr += 1
        elif esr >= early_stopping_round:
            break

    return  train_loss_list, valid_loss_list

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

    num_epoch = 50
    batch_size = 200
    num_k_fold = 8
    max_norm_grad = 1.0

    kwargs = {
        "num_heads" : 8,
        "hidden" : 64,
        "p" : 0.25,
        "alpha" : 0.1,
        "embedd_max_norm" : 1.0,
        "n_layers":4,
    }

    preds = []
    preds_test = []

    val_idxs = []

    k_fold = KFold(n_splits=num_k_fold, shuffle = True)
    start_time = time.time()

    from sklearn.model_selection import train_test_split

    df_train, df_test = train_test_split(train_df, test_size = 0.2, random_state=42, shuffle = True)

    test_loader = generate_dataloader(
        df_test,
        mode = 'submission', 
        test_size = None, 
        valid_size = None,
        batch_size = 128, 
        pred_col = None
    )

    for idx, (tr_idx, val_idx) in enumerate(k_fold.split(df_train)):
        
        pred = [] 
        pred_test = []

        tr_data = df_train.iloc[tr_idx]
        val_data = df_train.iloc[val_idx]

        train_loader, valid_loader = generate_dataloader_cv(df_train, 
            mode = 'train', batch_size = 128, 
            train_indices = tr_idx, valid_indices = val_idx,
            pred_col = 'Multi'
        )

        model = GATNet(**kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
        loss_fn = RMSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.95)

        root_dir = "./weights"
        best_pt = "cv_model_best_{}.pt".format(idx+1)
        last_pt = "cv_model_last_{}.pt".format(idx+1)

        train_cv(
            train_loader,
            valid_loader,
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            num_epoch,
            root_dir,
            best_pt,
            last_pt,
            max_norm_grad
        )

        model.load_state_dict(torch.load(os.path.join(root_dir, best_pt)), strict = False)

        valid_loss = valid_per_epoch(valid_loader,  model, optimizer, loss_fn, device)

        pred_test = inference(test_loader, model, device)
        preds_test.append(pred_test)
        
        del model, optimizer, loss_fn, scheduler, train_loader, valid_loader

        # memory cache init
        gc.collect()

        if idx == 0:
            end_time = time.time()
            dt = end_time - start_time
            dt_remain = dt * (num_k_fold - 1)
            
            h = int(dt_remain // 3600)
            m = int(dt_remain % 3600 // 60)
            s = int(dt_remain % 60)

            print("Extected time remained : {}h {}m {}s".format(h,m,s))

        print("metric : {:.3f}, {}-fold model training completed....[{}/{}]".format(valid_loss, idx+1, idx + 1, num_k_fold))

        # memory cache init
        gc.collect()

    preds_test = np.mean(preds_test, axis = 0)

    from sklearn.metrics import mean_squared_error

    test_loss = mean_squared_error(df_test[['Reorg_g', 'Reorg_ex']].values, preds_test, squared = False)
    print("test_loss : {:.3f}".format(test_loss))

    # inference
    preds_test = []

    k_fold = KFold(n_splits=num_k_fold, shuffle = True)
    start_time = time.time()

    submission_loader = generate_dataloader(
        test_df,
        mode = 'submission', 
        test_size = None, 
        valid_size = None,
        batch_size = 32, 
        pred_col = None
    )

    for idx, (tr_idx, val_idx) in enumerate(k_fold.split(train_df)):
        
        pred_test = []

        train_loader, valid_loader = generate_dataloader_cv(train_df, 
            mode = 'train', batch_size = 128, 
            train_indices = tr_idx, valid_indices = val_idx,
            pred_col = 'Multi'
        )

        model = GATNet(**kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
        loss_fn = RMSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.95)

        root_dir = "./weights"
        best_pt = "cv_model_best_{}.pt".format(idx+1)
        last_pt = "cv_model_last_{}.pt".format(idx+1)

        train_cv(
            train_loader,
            valid_loader,
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            num_epoch,
            root_dir,
            best_pt,
            last_pt,
            max_norm_grad
        )

        model.load_state_dict(torch.load(os.path.join(root_dir, best_pt)), strict = False)

        valid_loss = valid_per_epoch(valid_loader,  model, optimizer, loss_fn, device)

        pred_test = inference(submission_loader, model, device)
        preds_test.append(pred_test)
        
        del model, optimizer, loss_fn, scheduler, train_loader, valid_loader

        # memory cache init
        gc.collect()

        if idx == 0:
            end_time = time.time()
            dt = end_time - start_time
            dt_remain = dt * (num_k_fold - 1)
            
            h = int(dt_remain // 3600)
            m = int(dt_remain % 3600 // 60)
            s = int(dt_remain % 60)

            print("Extected time remained : {}h {}m {}s".format(h,m,s))

        print("metric : {:.3f}, {}-fold model training completed....[{}/{}]".format(valid_loss, idx+1, idx + 1, num_k_fold))

        # memory cache init
        gc.collect()

    preds_test = np.mean(preds_test, axis = 0)

    submission = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))
    submission.loc[:, ["Reorg_g", "Reorg_ex"]] =  preds_test
    submission.to_csv("./result/submission_GATNet_cv.csv", index = False)
