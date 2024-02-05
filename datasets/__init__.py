import torch.utils.data
import torchvision

def build_dataset(image_set, args):
    if args.dataset_file == 'bsi_panoptic':
        from .bsi_panoptic import build as build_bsi_panoptic
        return build_bsi_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
