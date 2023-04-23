import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR ,CyclicLR, ExponentialLR

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion


# newly imported
from datetime import datetime
import wandb
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model_final"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    # set directory for saving test results 
    now = datetime.now()
    folder_name = args.name +"_" +now.strftime('%Y-%m-%d-%H:%M:%S')
    save_dir = increment_path(os.path.join(model_dir,folder_name))
    # save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    # dataset_module = getattr(import_module("dataset_multilabel"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    # transform_module = getattr(import_module("dataset_multilabel"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model_final"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    # model = load_model('./model/2023-04-20-09:09:06', num_classes, device).to(device)

    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    #scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.95)
    #scheduler00 = CosineAnnealingWarmRestarts(optimizer, T_0=9, T_mult=2, eta_min=0, last_epoch=-1)
    #scheduler2 = CosineAnnealingLR(optimizer,T_max=10, eta_min=0, last_epoch=-1)
    scheduler00 = ExponentialLR(optimizer, gamma=0.95)
    #scheduler00 = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.003, step_size_up=5, step_size_down=5, mode='triangular2', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.95, last_epoch=-1)


    # train your model for one epoch

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # -- wandb initialize with configuration
    config={
        "epochs": args.epochs, 
        "batch_size": args.batch_size,
        "learning_rate" : args.lr,
        "architecture" : args.model
    }
    wandb.init(project="naver_boostcamp_AI_Tech_Level1", config=config,name=args.name)

    best_val_acc = 0
    best_val_loss = np.inf
    paitence = 0
    early_stop = args.epochs//5
    print(early_stop)
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0

        # Multi Label Classification
        # mask_matches = 0
        # gender_matches = 0
        # age_matches = 0        
        for idx, train_batch in enumerate(train_loader):
            # inputs, labels = train_batch
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # optimizer.zero_grad()

            # outs = model(inputs)
            # preds = torch.argmax(outs, dim=-1)
            # loss = criterion(outs, labels)

            # loss.backward()
            # optimizer.step()

            # loss_value += loss.item()
            # matches += (preds == labels).sum().item()

            # Multi Label Classification
            inputs, (mask_labels, gender_labels, age_labels) = train_batch
            inputs = inputs.to(device)
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            (mask_outs, gender_outs, age_outs) = torch.split(outs, [3,2,3], dim=1)
            mask_preds = torch.argmax(mask_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            age_preds = torch.argmax(age_outs, dim=-1)

            mask_loss = criterion(mask_outs, mask_labels)
            gender_loss = criterion(gender_outs, gender_labels)
            age_loss = criterion(age_outs, age_labels)

            loss = mask_loss + gender_loss + age_loss
            # loss = mask_loss + gender_loss + 1.5*age_loss

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (mask_preds == mask_labels).sum().item()
            matches += (gender_preds == gender_labels).sum().item()
            matches += (age_preds == age_labels).sum().item()
            # mask_matches += (mask_preds == mask_labels).sum().item()
            # gender_matches += (gender_preds == gender_labels).sum().item()
            # age_matches += (age_preds == age_labels).sum().item()

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                # train_acc = matches / args.batch_size / args.log_interval
                train_acc = (matches / args.batch_size / args.log_interval) / 3 # Multi Label Classification
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
        
        # logging wandb train phase
        wandb.log({
            'Train acc': train_acc, 
            'Train loss': train_loss
        })

        # scheduler.step()
        scheduler00.step()
        # if epoch < args.lr_decay_step:
        #     scheduler1.step()
        # else:
        #     scheduler2.step()


        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                # inputs, labels = val_batch
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # outs = model(inputs)
                # preds = torch.argmax(outs, dim=-1)

                # loss_item = criterion(outs, labels).item()
                # acc_item = (labels == preds).sum().item()
                # val_loss_items.append(loss_item)
                # val_acc_items.append(acc_item)

                # Multi Label Classification
                inputs, (mask_labels, gender_labels, age_labels) = val_batch
                inputs = inputs.to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)

                outs = model(inputs)
                (mask_outs, gender_outs, age_outs) = torch.split(outs, [3,2,3], dim=1)
                mask_preds = torch.argmax(mask_outs, dim=-1)
                gender_preds = torch.argmax(gender_outs, dim=-1)
                age_preds = torch.argmax(age_outs, dim=-1)

                mask_loss_item = criterion(mask_outs, mask_labels).item()
                gender_loss_item = criterion(gender_outs, gender_labels).item()
                age_loss_item = criterion(age_outs, age_labels).item()
                loss_item = mask_loss_item + gender_loss_item + age_loss_item
                # loss_item = mask_loss_item + gender_loss_item + 1.5*age_loss_item
                val_loss_items.append(loss_item)

                mask_acc_item = (mask_preds == mask_labels).sum().item()
                gender_acc_item = (gender_preds == gender_labels).sum().item()
                age_acc_item = (age_preds == age_labels).sum().item()
                acc_item = mask_acc_item + gender_acc_item + age_acc_item
                val_acc_items.append(acc_item)
                # val_acc_items.append(mask_acc_item)
                # val_acc_items.append(gender_acc_item)
                # val_acc_items.append(age_acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    # figure = grid_image(
                    #     inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    # )
                    figure = grid_image(
                        inputs_np, mask_labels, mask_preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
                    figure = grid_image(
                        inputs_np, gender_labels, gender_preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
                    figure = grid_image(
                        inputs_np, age_labels, age_preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )


            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set) / 3
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
                paitence=0
            else:
                print(f'no improvement for {paitence} epochs')
                print(f"Best val accuracy : {best_val_acc:4.2%}")
                print(f'current val accuracy : {val_acc:4.2%}')
                paitence+=1
                if paitence >= early_stop:
                    print('@@@@@ Early stopping @@@@@ !!!!!!! ')
                    break
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()

            # logging wandb valid phase
            wandb.log({
                'Valid acc': val_acc, 
                'Valid loss': val_loss
            })
    logger.close()
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ViT', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=3e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.3, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=1, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--config', type=str, help='path to the configuration file')
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
