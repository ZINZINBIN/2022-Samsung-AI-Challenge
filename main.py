import os
import torch
import pandas as pd
from src.preprocessing import generate_dataloader
from src.loss import RMSELoss
from src.model import ChebNet
from src.train import train
from src.utils import plot_training_curve
from src.evaluate import evaluate
from src.inference import inference

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:1" 
else:
    device = 'cpu'

# cache init
torch.cuda.empty_cache()

if __name__ == "__main__":
    PATH = "./dataset/"

    train_df = pd.read_csv(os.path.join(PATH, "train_set.csv"))
    test_df = pd.read_csv(os.path.join(PATH, "test_set.csv"))

    num_epoch = 128
    verbose = 8
    batch_size = 128

    k = 4
    hidden = 64
    alpha = 0.1
    embedd_max_norm = 1.0 
    n_layers = 6

    train_loader_g, valid_loader_g, test_loader_g = generate_dataloader(
        train_df,
        mode = 'train', 
        test_size = 0.2, 
        valid_size = 0.2,
        batch_size = batch_size, 
        pred_col = 'Reorg_g'
    )

    train_loader_ex, valid_loader_ex, test_loader_ex = generate_dataloader(
        train_df,
        mode = 'train', 
        test_size = 0.2, 
        valid_size = 0.2,
        batch_size = batch_size, 
        pred_col = 'Reorg_ex'
    )

    # for Reorg_g
    model_g = ChebNet(
        k = k, 
        hidden = hidden, 
        alpha = alpha, 
        embedd_max_norm = embedd_max_norm, 
        n_layers = n_layers
    )

    model_g.to(device)

    optimizer = torch.optim.AdamW(model_g.parameters(), lr = 1e-3)
    loss_fn = RMSELoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 8, T_mult=2)


    train_loss, valid_loss = train(
        train_loader_g,
        valid_loader_g,
        model_g,
        optimizer,
        scheduler,
        loss_fn,
        device,
        num_epoch = num_epoch,
        verbose = verbose,
        root_dir = "./weights",
        best_pt = "chebnet_g_best.pt",
        last_pt = "chebnet_g_last.pt",
        max_norm_grad = 1.0
    )

    plot_training_curve("./result/chebnet_g_learning_curve.png", train_loss, valid_loss)

    model_g.load_state_dict(torch.load("./weights/chebnet_g_best.pt"), strict = False)
    test_loss = evaluate(test_loader_g, model_g, optimizer, loss_fn, device)

    # for Reorg_ex
    
    model_ex = ChebNet(
        k = k, 
        hidden = hidden, 
        alpha = alpha, 
        embedd_max_norm = embedd_max_norm, 
        n_layers = n_layers
    )

    model_ex.to(device)

    optimizer = torch.optim.AdamW(model_ex.parameters(), lr = 1e-3)
    loss_fn = RMSELoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 8, T_mult=2)

    train_loss, valid_loss = train(
        train_loader_ex,
        valid_loader_ex,
        model_ex,
        optimizer,
        scheduler,
        loss_fn,
        device,
        num_epoch = num_epoch,
        verbose = verbose,
        root_dir = "./weights",
        best_pt = "chebnet_ex_best.pt",
        last_pt = "chebnet_ex_last.pt",
        max_norm_grad = 1.0
    )

    plot_training_curve("./result/chebnet_ex_learning_curve.png", train_loss, valid_loss)

    model_ex.load_state_dict(torch.load("./weights/chebnet_ex_best.pt"), strict = False)
    test_loss = evaluate(test_loader_ex, model_ex, optimizer, loss_fn, device)

    # inference
    submission_loader = generate_dataloader(
        test_df,
        mode = 'submission', 
        test_size = None, 
        valid_size = None,
        batch_size = 128, 
        pred_col = None
    )

    pred_g = inference(submission_loader, model_g, device)
    pred_ex = inference(submission_loader, model_ex, device)

    submission = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))
    submission.loc[:, ["Reorg_g"]] =  pred_g
    submission.loc[:, ["Reorg_ex"]] =  pred_ex

    submission.to_csv("./result/submission_chebnet.csv", index = False)