import torch
import numpy as np

import data_config
from datasets.SkinCancerData import Fitzpatrick
from torch.utils.data import DataLoader,WeightedRandomSampler

from sklearn.utils.class_weight import compute_class_weight

from torchvision import utils

def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])  

def get_dataloaders(args):
    
    data_name = args.data_name
    data_conf = data_config.DataConfig().get_data_config(data_name=data_name)

    root_dir = data_conf.root_dir
    split = args.split
    split_val = args.split_val

    if data_name == 'Fitzpatrick17k':
        training_set = Fitzpatrick(root_dir=root_dir, img_size=args.img_size, data_name=data_name, is_train=True, split=split)
        val_set = Fitzpatrick(root_dir=root_dir, img_size=args.img_size, data_name=data_name, is_train=False, split=split_val)
    elif data_name == 'test':
        training_set = Fitzpatrick(root_dir=root_dir, img_size=args.img_size, data_name=data_name, is_train=True, split=split)
        val_set = Fitzpatrick(root_dir=root_dir, img_size=args.img_size, data_name=data_name, is_train=True, split=split)
    elif data_name == 'Fitzpatrick17k_balanced':
        training_set = Fitzpatrick(root_dir=root_dir, img_size=args.img_size, data_name=data_name, is_train=True, split=split)
        val_set = Fitzpatrick(root_dir=root_dir, img_size=args.img_size, data_name=data_name, is_train=False, split=split_val)
    else:
        raise TypeError('%s has not defined' % data_name)
    
    datasets = {'train': training_set, 'val': val_set}

    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size
                                 , num_workers=args.num_workers) 
                                 for x in ['train', 'val']}
    
    return dataloaders

def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis

def get_weighted_sampler(dataset,split):
    """
    Return a WeightedRandomSampler that balances the binary classes
    (benign vs malignant) derived from a three‑partition label column.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Must expose a pandas DataFrame in `dataset.df` that has a column
        called ``three_partition_label`` with the values:
            * 'benign'
            * 'non-neoplastic'   → treated as benign
            * 'malignant'

    Returns
    -------
    torch.utils.data.WeightedRandomSampler
        Sampler that samples each example with probability
        proportional to the inverse class frequency.
    """
    if split == 'val':
        return None
    three_labels = dataset.df['three_partition_label']

    label_to_idx = {'benign': 0, 'malignant': 1,'non-neoplastic':2}
    numeric = three_labels.map(label_to_idx).values.astype(np.int64)


    class_counts = np.bincount(numeric)          # shape: [2]


    class_weights = 1.0 / class_counts.astype(np.float32)



    sample_weights = class_weights[numeric]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),      # total number of samples to draw
        replacement=True
    )
    return sampler
