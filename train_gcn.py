import os
import torch
import pandas as pd
from src.preprocessing import generate_dataloader
from src.loss import RMSELoss
from src.model import GConvNet
from src.train import train
from src.utils import plot_training_curve
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

    num_epoch = 128
    verbose = 8
    batch_size = 128

    hidden = 128
    alpha = 0.1
    n_layers = 4
    embedd_max_norm = 1.0
    agg = 'add'

    lr = 1e-3
    gamma = 0.95
    step_size = num_epoch // 4
    output_dim = 2

    train_loader, valid_loader, _ = generate_dataloader(
        train_df,
        mode = 'train',
        test_size = None,
        valid_size = 0.2,
        batch_size = batch_size,
        pred_col = 'Multi'
    )

    model = GConvNet(
        output_dim = output_dim,
        hidden = hidden, 
        alpha = alpha, 
        n_layers = n_layers,
        embedd_max_norm = embedd_max_norm, 
        agg = agg
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    loss_fn = RMSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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
        best_pt = "gcn_best.pt",
        last_pt = "gcn_last.pt",
        max_norm_grad = 1.0
    )

    plot_training_curve("./result/GCN_learning_curve.png", train_loss, valid_loss)
    model.load_state_dict(torch.load("./weights/gcn_best.pt"), strict = False)

    # inference
    submission_loader = generate_dataloader(
        test_df,
        mode = 'submission', 
        test_size = None, 
        valid_size = None,
        batch_size = batch_size, 
        pred_col = None
    )

    pred = inference(submission_loader, model, device)

    submission = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))
    submission.loc[:, ["Reorg_g", "Reorg_ex"]] =  pred
    submission.to_csv("./result/submission_gcn.csv", index = False)