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
    opt = 'options/train/train_BDNet.yml'
    opt = option.parse(opt, is_train=True)
    log_dir = '../tb_logger/' + opt['name']
    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None:
        util.mkdir_and_rename(
            opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key and not path ==None))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # tensorboard loggercur_lr
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        version = float(torch.__version__[0:3])
        if version >= 1.1:  # PyTorch 1.1
            from torch.utils.tensorboard import SummaryWriter
        else:
            logger.info(
                'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
            from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir=log_dir)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            train_sampler = None
            train_loader_syn = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
        elif phase == 'train1':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            train_sampler = None
            train_loader_real = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
        elif phase == 'val':
            # pdb.set_trace()
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader_syn is not None
    assert train_loader_real is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    total_epochs = 90
    start_epoch = 0
    size_dataset_syn = len(train_loader_syn)
    size_dataset_real = len(train_loader_real)
    size_all = size_dataset_syn if size_dataset_syn >= size_dataset_real else size_dataset_real
    Trainlist0 = np.zeros(size_all, dtype=np.float32)
    Trainlist1 = np.zeros(size_all, dtype=np.float32)
    
    train_loader_syn_iter = iter(train_loader_syn)
    train_loader_real_iter = iter(train_loader_real)
    for epoch in range(start_epoch, total_epochs + 1):
        #### update learning rate
        
        syn_count = 0
        real_count = 0
        if epoch>=0 and epoch<30:
            rang = 2
        elif epoch>=30 and epoch<60:
            rang = 5
        else:
            rang = 3
        
        for num in range(size_all):
            flag_num = np.random.randint(rang)
            if flag_num > 0:            ### Synthetic 
                try:
                    st=time.time()
                    train_data = train_loader_syn_iter.next()
                    syn_count += 1
                    #### training
                    model.feed_data(train_data)
                    key = train_data['key']
                    model.optimize_parameters(global_epoch=epoch,flag='syn')
                    cur_lr = model.get_current_learning_rate()[0]
                    logs = model.get_current_log()
                    loss = logs['l_pix']
                    Trainlist0[num] = loss
                    show_loss = np.mean(Trainlist0[np.where(Trainlist0)])
                    print('tik 1 epoch:%03d, step:%d/%d, loss:%.4f, lr:%.8f,time:%.3f'%(epoch,syn_count,size_dataset_syn,show_loss,cur_lr,time.time()-st))
                except StopIteration: 
                    train_loader_syn_iter = iter(train_loader_syn)
                    syn_count = 0
                    st=time.time()
                    train_data = train_loader_syn_iter.next()
                    syn_count += 1
                    #### training
                    model.feed_data(train_data)
                    key = train_data['key']
                    model.optimize_parameters(global_epoch=epoch,flag='syn')
                    cur_lr = model.get_current_learning_rate()[0]
                    logs = model.get_current_log()
                    loss = logs['l_pix']
                    Trainlist0[num] = loss
                    show_loss = np.mean(Trainlist0[np.where(Trainlist0)])
                    print('tik 1 epoch:%03d, step:%d/%d, loss:%.4f, lr:%.8f, time:%.3f'%(epoch,syn_count,size_dataset_syn,show_loss,cur_lr,time.time()-st))
            else: 
                try:
                    st=time.time()
                    train_data = train_loader_real_iter.next()
                    real_count += 1
                    key = train_data['key']
                    model.feed_data(train_data)
                    model.optimize_parameters(global_epoch=epoch,flag='real')            
                    cur_lr = model.get_current_learning_rate()[0]            
                    logs = model.get_current_log()
                    loss = logs['l_pix']
                    Trainlist1[num] = loss
                    show_loss = np.mean(Trainlist1[np.where(Trainlist1)])
                    print('tik 2 epoch:%03d, step:%d/%d, loss:%.4f, lr:%.8f,time:%.3f'%(epoch,real_count,size_dataset_real,show_loss,cur_lr,time.time()-st))
                except StopIteration: 
                    train_loader_real_iter = iter(train_loader_real)  
                    real_count = 0
                    st=time.time()
                    train_data = train_loader_real_iter.next()
                    real_count += 1
                    #### training
                    model.feed_data(train_data)
                    key = train_data['key']
                    model.optimize_parameters(global_epoch=epoch,flag='real')
                    cur_lr = model.get_current_learning_rate()[0]
                    logs = model.get_current_log()
                    loss = logs['l_pix']
                    Trainlist0[num] = loss
                    show_loss = np.mean(Trainlist0[np.where(Trainlist0)])
                    print('tik 2 epoch:%03d, step:%d/%d, loss:%.4f, lr:%.8f, time:%.3f'%(epoch,real_count,size_dataset_real,show_loss,cur_lr,time.time()-st))
                
        model.update_learning_rate(start_epoch, warmup_iter=opt['train']['warmup_iter'])    
        
        #### validation
        if opt['datasets'].get('val', None) and epoch % 2 == 0:
            #pbar = util.ProgressBar(len(val_loader))
            psnr_rlt = {}  # with border and center frames
            psnr_rlt_avg = {}
            psnr_total_avg = 0.
            ssim_rlt = {}  # with border and center frames
            ssim_rlt_avg = {}
            ssim_total_avg = 0.
            save_path = "%s/Real_static/%04d"%(log_dir,epoch)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            model.save('%02d'%(epoch))
            model.save_training_state(epoch, epoch)
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

        
    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()
    

if __name__ == '__main__':
    main()
