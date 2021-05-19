import sys
import os
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt" # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import time
import datetime
import argparse
import copy

import numpy as np
import pandas as pd

from dataloaders.StreamingDatasets import StreamingGeospatialDataset

import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models
import utils

NUM_WORKERS = 4
NUM_CHIPS_PER_TILE = 100
CHIP_SIZE = 256

parser = argparse.ArgumentParser(description='DFC2021 baseline training script')
parser.add_argument('--input_fn', type=str, required=True,  help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('--output_dir', type=str, required=True,  help='The path to a directory to store model checkpoints.')
parser.add_argument('--overwrite', action="store_true",  help='Flag for overwriting `output_dir` if that directory already exists.')
parser.add_argument('--save_most_recent', action="store_true",  help='Flag for saving the most recent version of the model during training.')
parser.add_argument('--model', default='unet',
    choices=(
        'unet',
        'fcn',
        'hrnet'
    ),
    help='Model to use'
)

## Training arguments
parser.add_argument('--gpu', type=int, default=0, help='The ID of the GPU to use')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for training')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--seed', type=int, default=0, help='Random seed to pass to numpy and torch')
args = parser.parse_args()

def image_transforms(img, group):
    if group == 0:
        img = (img - utils.NAIP_2013_MEANS) / utils.NAIP_2013_STDS
    elif group == 1:
        img = (img - utils.NAIP_2017_MEANS) / utils.NAIP_2017_STDS
    else:
        raise ValueError("group not recognized")
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img

def label_transforms(labels, group):
    labels = utils.NLCD_CLASS_TO_IDX_MAP[labels]
    labels = torch.from_numpy(labels)
    return labels

def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)


def main():
    print("Starting DFC2021 baseline training script at %s" % (str(datetime.datetime.now())))


    #-------------------
    # Setup
    #-------------------
    assert os.path.exists(args.input_fn)

    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)):
        if args.overwrite:
            print("WARNING! The output directory, %s, already exists, we might overwrite data in it!" % (args.output_dir))
        else:
            print("The output directory, %s, already exists and isn't empty. We don't want to overwrite and existing results, exiting..." % (args.output_dir))
            return
    else:
        print("The output directory doesn't exist or is empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
    else:
        print("WARNING! Torch is reporting that CUDA isn't available, exiting...")
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    #-------------------
    # Load input data
    #-------------------
    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    groups = input_dataframe["group"].values

    dataset = StreamingGeospatialDataset(
        imagery_fns=image_fns, label_fns=label_fns, groups=groups, chip_size=CHIP_SIZE, num_chips_per_tile=NUM_CHIPS_PER_TILE, windowed_sampling=False, verbose=False,
        image_transform=image_transforms, label_transform=label_transforms, nodata_check=nodata_check
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    num_training_batches_per_epoch = int(len(image_fns) * NUM_CHIPS_PER_TILE / args.batch_size)
    print("We will be training with %d batches per epoch" % (num_training_batches_per_epoch))


    #-------------------
    # Setup training
    #-------------------
    if args.model == "unet":
        model = models.get_unet()
    elif args.model == "fcn":
        model = models.get_fcn()
    elif args.model == "hrnet":
        model = models.get_hrnet()
    else:
        raise ValueError("Invalid model")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    print("Model has %d parameters" % (utils.count_parameters(model)))


    #-------------------
    # Model training
    #-------------------
    training_task_losses = []
    num_times_lr_dropped = 0 
    model_checkpoints = []
    temp_model_fn = os.path.join(args.output_dir, "most_recent_model.pt")

    for epoch in range(args.num_epochs):
        lr = utils.get_lr(optimizer)

        training_losses = utils.fit(
            model,
            device,
            dataloader,
            num_training_batches_per_epoch,
            optimizer,
            criterion,
            epoch,
        )
        scheduler.step(training_losses[0])

        model_checkpoints.append(copy.deepcopy(model.state_dict()))
        if args.save_most_recent:
            torch.save(model.state_dict(), temp_model_fn)

        if utils.get_lr(optimizer) < lr:
            num_times_lr_dropped += 1
            print("")
            print("Learning rate dropped")
            print("")
            
        training_task_losses.append(training_losses[0])
            
        if num_times_lr_dropped == 4:
            break


    #-------------------
    # Save everything
    #-------------------
    save_obj = {
        'args': args,
        'training_task_losses': training_task_losses,
        "checkpoints": model_checkpoints
    }

    save_obj_fn = "results.pt"
    with open(os.path.join(args.output_dir, save_obj_fn), 'wb') as f:
        torch.save(save_obj, f)

if __name__ == "__main__":
    main()