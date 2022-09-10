"""
Rather simple example training script for the ATP probe, without any bells and whistles, to help illustrate ATP usage
and integration into a broader training pipeline. Feel free to modify heavily to fit training needs.
"""

from atp import ATPSelectorModel, ATPConfig, atp_downstream_task_forward
import data
import numpy as np
import random

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate

def log(message, logger=None):
    """
    Placeholder log function; replace with your loggers/apis of choice (e.g. wandb, etc.)
    """
    if logger is not None: raise NotImplemented("implement your own logger")
    print(message)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def process_batch(batch, set_to_device=None, replace_empty_with_none=False):
    if set_to_device is not None:
        batch = [_b.to(set_to_device) if torch.is_tensor(_b) else _b for _b in batch]
    if replace_empty_with_none:
        batch = [_b if len(_b) > 0 else None for _b in batch]
    return batch

def main(args):
    seed_everything(args.seed)
    # create ATPSelectorModel from model hyperparameters
    config = ATPConfig.from_args(args)
    device = torch.device("gpu:0" if args.gpus > 0 else "cpu")
    atp_model = ATPSelectorModel(config, **vars(args)).to(device)

    # create datasets and dataloaders
    dset_train = data.VideoLanguageDataset(args, split="train")
    dset_val = data.VideoLanguageDataset(args, split="val")
    dldr_train = torch.utils.data.DataLoader(dset_train,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)
    dldr_val   = torch.utils.data.DataLoader(dset_val,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)

    # create optimizer
    if args.wd > 0.0:
        optim = torch.optim.AdamW(atp_model.parameters(), 
                                  lr=args.lr, weight_decay=args.wd)
    else:
        optim = torch.optim.Adam(atp_model.parameters(), lr=args.lr)

    # simple training loop (for illustrative purposes)
    for epoch_i in range(args.epochs):
        # train epoch
        atp_model.train()
        for i, batch in enumerate(dldr_train):
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
            # refactored the "forward pass" here into an example code snippet in atp.py; feel free to modify/replace here!
            loss, accs, selected_frames, y_pred, out_masks = atp_downstream_task_forward(atp_model, batch)
            atp_model.zero_grad(set_to_none=True)
            loss.backward()
            # do logging stuff with accs, selected_frames, masks, etc. For example:
            log(f"train: epoch{epoch_i}, iter{i}: loss = {loss.item()}, acc = {accs.mean().item()}")
            if args.grad_clip_val > 0.0:
                nn.utils.clip_grad_norm_(atp_model.parameters(), args.grad_clip_val)
            optim.step()

        # val epoch
        atp_model.eval()
        all_val_accs = []
        for i, batch in enumerate(dldr_val):
            with torch.no_grad():
                batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
                loss, accs, selected_frames, y_pred, out_masks = atp_downstream_task_forward(atp_model, batch)
                all_val_accs.append(accs)
        overall_acc = torch.cat(all_val_accs).mean().item()
        log(f"val: epoch{epoch_i}: overall_acc = {overall_acc}")
        # do additional checkpointing of atp_model.parameters() here, with whatever preferred API.
    return 0


# parse args and invoke main()
def add_bool_arg(parser, arg_name, default=True):
    parser.add_argument(f'--{arg_name}', action='store_true')
    parser.add_argument(f'--{arg_name}', dest=f'{arg_name}', action='store_false')
    parser.set_defaults(**{arg_name : default})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parser for ATP example training script.")
    
    # Training hyperparameters
    parser.add_argument('--batch_size', default=512, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--wd', default=0.0, type=float, help="weight decay")
    parser.add_argument('--epochs', default=1000, type=int, help="number of training epochs")
    parser.add_argument('--grad_clip_val', default=1.0, type=float, help="gradient clip, must be set > 0 to enable")
    parser.add_argument('--gpus', default=1, type=int)  # NOTE: current script is set-up for single-gpu training only.
    parser.add_argument('--num_workers', default=0, type=int, help="number of dataset workers")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    
    # ATP model hyperparameters (for more help/details, see ATPConfig)
    parser.add_argument('--n_layers', default=2, type=int, help="see ATPConfig")
    parser.add_argument('--n_heads', default=2, type=int, help="see ATPConfig")
    parser.add_argument('--d_model', default=128, type=int, help="see ATPConfig")
    parser.add_argument('--d_model_ff', default=128, type=int, help="see ATPConfig")
    parser.add_argument('--enc_dropout', default=0.1, type=float, help="see ATPConfig")
    add_bool_arg(parser, "use_text_query", True, help="see ATPConfig")
    add_bool_arg(parser, "use_text_cands", True, help="see ATPConfig")
    parser.add_argument('--enc_dropout', default=0.1, type=float, help="see ATPConfig")
    add_bool_arg(parser, "use_ste", True, help="see ATPConfig")
    parser.add_argument('--sel_dropout', default=0.0, type=float, help="see ATPConfig")
    parser.add_argument('--d_input', default=512, type=int, help="see ATPConfig")
    
    # I/O and data parameters
    parser.add_argument('--data_path', type=str, required=True, help="path to data; see data.py and data/")
    parser.add_argument('--n_frames', default=8, type=int, help="number of frames sampled for input; see data.py")
    
    args = parser.parse_args()
    main(args)

