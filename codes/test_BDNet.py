import os
import math
import random
import logging,time
import data.util as datautil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import cv2
import numpy as np

def nor(x):
    y = (x-x.min())/(x.max()-x.min())
    return y
    
def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():

    #### options
    opt = 'options/test/test_BDNet.yml'
    opt = option.parse(opt, is_train=True)
    log_dir = '../tb_logger/test_' + opt['name']
    #### distributed training settings
    opt['dist'] = False
    print('Disabled distributed training.')



    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            # pdb.set_trace()
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)

    #### create model
    model = create_model(opt)




    
        
    psnr_rlt = {}  # with border and center frames
    psnr_total_avg = 0.
    ssim_rlt = {}  # with border and center frames
    ssim_total_avg = 0.
    save_path = "%s/Real_static"%(log_dir)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    is_test_gt = opt['is_test_gt']
    
    #if epoch % 10 == 0:
    for idx, val_data in enumerate(val_loader):
        
            
        folder = val_data['folder'][0]
        model.feed_data(val_data,need_GT=is_test_gt)
        model.test(flag='real')
        visuals = model.get_current_visuals(need_GT=is_test_gt)
        lq_img = visuals['LQ'][2,:,:,:].permute(1,2,0).numpy()
        lq_img = datautil.demosaic(lq_img)
        rlt_img = util.tensor2img(visuals['rlt'],out_type=np.float32)  # uint8
        gt_img = util.tensor2img(visuals['GT'],out_type=np.float32) if is_test_gt == True else None # uint8
        
        out_img = np.concatenate([lq_img,rlt_img,gt_img],1) if is_test_gt == True else np.concatenate([lq_img,rlt_img],1) # uint8
        wb = val_data['wb'][0].unsqueeze(0).numpy()
        out_img = out_img*wb
        out_img = (np.clip(out_img,0.0,1.0-1e-4)+1e-4)**(1.0/2.2)
        out_img = np.uint8(out_img*255.0)
        
        cv2.imwrite("%s/%05d.jpg"%(save_path,idx),
            out_img[:,:,::-1],[int(cv2.IMWRITE_JPEG_QUALITY),100])
        
        # calculate PSNR
        if is_test_gt == True:
            psnr = util.calculate_psnr(np.uint8(np.clip(rlt_img,0.0,1.0)*255.0), np.uint8(np.clip(gt_img,0.0,1.0)*255.0))
            if math.isinf(psnr) == False:
                psnr_rlt[folder] = psnr
                print('idx = %04d, psnr = %.4f'%(idx,psnr))
            ssim = util.calculate_ssim(np.uint8(np.clip(rlt_img,0.0,1.0)*255.0), np.uint8(np.clip(gt_img,0.0,1.0)*255.0))
            if math.isinf(ssim) == False:
                ssim_rlt[folder] = ssim

        #pbar.update('Test {}_psnr={}'.format(folder,psnr))
    if is_test_gt == True:
        for k, v in psnr_rlt.items():
            psnr_total_avg += v
        for k, v in ssim_rlt.items():
            ssim_total_avg += v
        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            target=open("%s/score.txt"%(save_path),'w')
            target.write("psnr=%.4f\n"%(psnr_total_avg))
            target.write("ssim=%.4f\n"%(ssim_total_avg))
            target.close()

        

    

if __name__ == '__main__':
    main()
