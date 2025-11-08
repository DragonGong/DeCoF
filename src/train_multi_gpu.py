import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.data import Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import roc_auc_score, accuracy_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from detect import Detector
from utils.sam import SAM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_ddp():
    # 初始化进程组
    dist.init_process_group(backend="nccl")  # nccl for GPU
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def compute_accuracy(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def main(args):
    rank, local_rank = setup_ddp()
    is_main_process = (rank == 0)

    cfg = load_json(args.config)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image_size = cfg['image_size']
    batch_size = cfg['batch_size']
    n_frames = cfg['n_frames']
    subdatasets_name = cfg['subdatasets_name']
    n_epoch = cfg['epoch']
    trained_model_path = cfg['trained_model']

    # Model
    model = Detector().to(local_rank)
    if trained_model_path != "":
        checkpoint = torch.load(trained_model_path)
        model.net_all.load_state_dict(checkpoint["model"])
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Dataset & DataLoader with DistributedSampler
    train_dataset = Dataset(phase='Train', data_name=subdatasets_name, image_size=image_size, n_frames=n_frames)
    val_dataset = Dataset(phase='Val', data_name=subdatasets_name, image_size=image_size, n_frames=n_frames)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=20,  # reduced from 20 to avoid too many threads per process
        pin_memory=True,
        # drop_last=True  # optional, but helps with consistent batch size
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=20,
        pin_memory=True,
        # drop_last=False
    )

    # Optimizer & Scheduler
    optimizer = SAM(
        model.module.net_all.parameters(),
        torch.optim.SGD,
        lr=0.001,
        momentum=0.9
    )
    lr_scheduler = LinearDecayLR(optimizer, n_epoch, int(n_epoch * 3 / 4))

    criterion = nn.CrossEntropyLoss()

    if is_main_process:
        now = datetime.now()
        save_path = './output/{}_'.format(args.session_name) + \
                    os.path.splitext(os.path.basename(args.config))[0] + '_' + \
                    now.strftime("%m_%d_%H_%M_%S") + '/'
        os.makedirs(save_path + 'weights/', exist_ok=True)
        os.makedirs(save_path + 'logs/', exist_ok=True)
        logger = log(path=save_path + "logs/", file="losses.logs")
        print(f"Train loader length: {len(train_loader)}")
        print(f"Val loader length: {len(val_loader)}")
    else:
        save_path = None
        logger = None

    last_val_auc = 0.0
    weight_dict = {}
    n_weight = 20

    for epoch in range(n_epoch):
        # Set epoch for sampler (ensures different shuffling per epoch)
        train_sampler.set_epoch(epoch)

        model.train()
        pred_first_list, target_list = [], []
        train_loss = 0.0

        for step, data in enumerate(tqdm(train_loader, disable=(not is_main_process))):
            img = data['img'].to(local_rank, non_blocking=True).float()
            target = data['label'].to(local_rank, non_blocking=True).long()

            # --- SAM with two forward passes ---
            pred_first = None
            for i in range(2):
                pred_cls = model(img)  # forward
                if i == 0:
                    pred_first = pred_cls.detach()  # save first prediction (for metrics)

                loss_cls = criterion(pred_cls, target)
                loss = loss_cls

                optimizer.zero_grad()
                loss.backward()

                if i == 0:
                    optimizer.first_step(zero_grad=True)
                else:
                    optimizer.second_step(zero_grad=True)

            # Accumulate the first prediction for metric
            pred_first_list.extend(F.log_softmax(pred_first, dim=1).argmax(dim=1).cpu().numpy())
            target_list.extend(target.cpu().numpy())
            train_loss += loss_cls.item()

        lr_scheduler.step()

        # Gather metrics across all processes (optional but recommended for accuracy)
        # For simplicity, we only compute on each process and average (approximate)
        train_acc = accuracy_score(target_list, pred_first_list)
        avg_train_loss = train_loss / len(train_loader)

        # Validation (only compute on all ranks, but save/log from rank 0)
        model.eval()
        val_pred_scores, val_preds, val_targets = [], [], []

        with torch.no_grad():
            for step, data in enumerate(tqdm(val_loader, disable=(not is_main_process))):
                img = data['img'].to(local_rank, non_blocking=True).float()
                target = data['label'].to(local_rank, non_blocking=True).long()
                output = model(img)
                probs = F.softmax(output, dim=1)
                val_pred_scores.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(probs.argmax(dim=1).cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # Compute AUC and ACC (each process has a subset; for exact metric, gather all, but we approximate)
        try:
            val_auc = roc_auc_score(val_targets, val_pred_scores)
            val_acc = accuracy_score(val_targets, val_preds)
        except Exception as e:
            if is_main_process:
                print("AUC compute error:", e)
            val_auc = 0.0
            val_acc = 0.0

        # Only main process logs and saves
        if is_main_process:
            log_text = f"Epoch {epoch+1}/{n_epoch} | train loss: {avg_train_loss:.4f}, train acc: {train_acc:.4f}, val acc: {val_acc:.4f}, val auc: {val_auc:.4f}"
            logger.info(log_text)
            print(log_text)

            # Save top-k models by val_auc
            if len(weight_dict) < n_weight:
                save_model_path = os.path.join(save_path + 'weights/', f"{epoch+1}_{val_auc:.4f}_val.tar")
                weight_dict[save_model_path] = val_auc
                torch.save({
                    "model": model.module.net_all.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }, save_model_path)
                last_val_auc = min(weight_dict.values())
            elif val_auc >= last_val_auc:
                save_model_path = os.path.join(save_path + 'weights/', f"{epoch+1}_{val_auc:.4f}_val.tar")
                # Remove worst model
                worst_path = min(weight_dict, key=weight_dict.get)
                del weight_dict[worst_path]
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                # Save new one
                weight_dict[save_model_path] = val_auc
                torch.save({
                    "model": model.module.net_all.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }, save_model_path)
                last_val_auc = min(weight_dict.values())

    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session_name')
    args = parser.parse_args()

    main(args)