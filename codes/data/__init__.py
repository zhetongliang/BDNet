"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data
import pdb


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
        batch_size = dataset_opt['batch_size']
        shuffle = dataset_opt['use_shuffle']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
        
    if phase == 'train1':
        num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
        batch_size = dataset_opt['batch_size']
        shuffle = dataset_opt['use_shuffle']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
        
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration

    if mode == 'Synthetic_dynamic_dataset':
        from data.Denoising_dataset import Synthetic_dynamic_dataset as D
    elif mode == 'Burst_static_dataset':
        from data.Denoising_dataset import Burst_static_dataset as D
    elif mode == 'Real_static':
        from data.video_denoising_test_dataset import Real_static as D
    elif mode == 'Real_dynamic':
        from data.video_denoising_test_dataset import Real_dynamic as D
        # pdb.set_trace()
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
        
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
