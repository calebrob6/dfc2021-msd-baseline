import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

NAIP_2013_MEANS = np.array([117.00, 130.75, 122.50, 159.30])
NAIP_2013_STDS = np.array([38.16, 36.68, 24.30, 66.22])
NAIP_2017_MEANS = np.array([72.84,  86.83, 76.78, 130.82])
NAIP_2017_STDS = np.array([41.78, 34.66, 28.76, 58.95])
NLCD_CLASSES = [ 0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95] # 16 classes + 1 nodata class ("0"). Note that "12" is "Perennial Ice/Snow" and is not present in Maryland.

NLCD_CLASS_COLORMAP = { # Copied from the emebedded color table in the NLCD data files
    0:  (0, 0, 0, 255),
    11: (70, 107, 159, 255),
    12: (209, 222, 248, 255),
    21: (222, 197, 197, 255),
    22: (217, 146, 130, 255),
    23: (235, 0, 0, 255),
    24: (171, 0, 0, 255),
    31: (179, 172, 159, 255),
    41: (104, 171, 95, 255),
    42: (28, 95, 44, 255),
    43: (181, 197, 143, 255),
    52: (204, 184, 121, 255),
    71: (223, 223, 194, 255),
    81: (220, 217, 57, 255),
    82: (171, 108, 40, 255),
    90: (184, 217, 235, 255),
    95: (108, 159, 184, 255)
}

LC4_CLASS_COLORMAP = {
    0: (0, 0, 255, 255),
    1: (0, 128, 0, 255),
    2: (128, 255, 128, 255),
    3: (128, 96, 96, 255),
    4: (0, 0, 0, 255)
}


NLCD_IDX_COLORMAP = {
    idx: NLCD_CLASS_COLORMAP[c]
    for idx, c in enumerate(NLCD_CLASSES)
}

def get_nlcd_class_to_idx_map():
    nlcd_label_to_idx_map = []
    idx = 0
    for i in range(NLCD_CLASSES[-1]+1):
        if i in NLCD_CLASSES:
            nlcd_label_to_idx_map.append(idx)
            idx += 1
        else:
            nlcd_label_to_idx_map.append(0)
    nlcd_label_to_idx_map = np.array(nlcd_label_to_idx_map).astype(np.int64)
    return nlcd_label_to_idx_map

NLCD_CLASS_TO_IDX_MAP = get_nlcd_class_to_idx_map() # I do this computation on import for illustration (this could instead be a length 96 vector that is hardcoded here)


NLCD_IDX_TO_REDUCED_LC_MAP = np.array([
    4,#  0 No data 0
    0,#  1 Open Water
    4,#  2 Ice/Snow
    2,#  3 Developed Open Space
    3,#  4 Developed Low Intensity
    3,#  5 Developed Medium Intensity
    3,#  6 Developed High Intensity
    3,#  7 Barren Land
    1,#  8 Deciduous Forest
    1,#  9 Evergreen Forest
    1,# 10 Mixed Forest
    1,# 11 Shrub/Scrub
    2,# 12 Grassland/Herbaceous
    2,# 13 Pasture/Hay
    2,# 14 Cultivated Crops
    1,# 15 Woody Wetlands
    1,# 16 Emergent Herbaceious Wetlands
])

NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR = np.array([
    [0,0,0,0,1],#  0 No data 0
    [1,0,0,0,0],#  1 Open Water
    [0,0,0,0,1],#  2 Ice/Snow
    [0,0,0,0,0],#  3 Developed Open Space
    [0,0,0,0,0],#  4 Developed Low Intensity
    [0,0,0,1,0],#  5 Developed Medium Intensity
    [0,0,0,1,0],#  6 Developed High Intensity
    [0,0,0,0,0],#  7 Barren Land
    [0,1,0,0,0],#  8 Deciduous Forest
    [0,1,0,0,0],#  9 Evergreen Forest
    [0,1,0,0,0],# 10 Mixed Forest
    [0,1,0,0,0],# 11 Shrub/Scrub
    [0,0,1,0,0],# 12 Grassland/Herbaceous
    [0,0,1,0,0],# 13 Pasture/Hay
    [0,0,1,0,0],# 14 Cultivated Crops
    [0,1,0,0,0],# 15 Woody Wetlands
    [0,1,0,0,0],# 16 Emergent Herbaceious Wetlands
])

class Timer():
    '''A wrapper class for printing out what is running and how long it took.
    
    Use as:
    ```
    with utils.Timer("running stuff"):
        # do stuff
    ```

    This will output:
    ```
    Starting 'running stuff'
    # any output from 'running stuff'
    Finished 'running stuff' in 12.45 seconds
    ```
    '''
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.tic = float(time.time())
        print("Starting '%s'" % (self.message))

    def __exit__(self, type, value, traceback):
        print("Finished '%s' in %0.4f seconds" % (self.message, time.time() - self.tic))


def fit(model, device, data_loader, num_batches, optimizer, criterion, epoch, memo=''):
    model.train()
    
    losses = []
    tic = time.time()
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader), total=num_batches, file=sys.stdout):
        data = data.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)
    
    print('[{}] Training Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
        memo, epoch, time.time()-tic, avg_loss), end=""
    )
    print("")
    
    return [avg_loss]

def evaluate(model, device, data_loader, num_batches, criterion, epoch, memo=''):
    model.eval()
    
    losses = []
    tic = time.time()
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader), total=num_batches, file=sys.stdout):
        data = data.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
    
    avg_loss = np.mean(losses)

    print('[{}] Validation Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
        memo, epoch, time.time()-tic, avg_loss), end=""
    )
    print("")
    
    return [avg_loss]

def score(model, device, data_loader, num_batches):
    model.eval()

    num_classes = model.module.segmentation_head[0].out_channels
    num_samples = len(data_loader.dataset)
    predictions = np.zeros((num_samples, num_classes), dtype=np.float32)
    idx = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            output = F.softmax(model(data))
        batch_size = data.shape[0]
        predictions[idx:idx+batch_size] = output.cpu().numpy()
        idx += batch_size
    return predictions

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)