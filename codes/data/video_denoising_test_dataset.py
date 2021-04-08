import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import pdb,random
import numpy as np
import cv2
import lmdb,glob
import scipy.io as scio



class Real_dynamic(data.Dataset):

    def __init__(self, opt):
        super(Real_dynamic, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
        # GS: No need LR
        #self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': []}   ## They are flattened
        #### Generate data info and cache data
        
        # subfolders_LQ = util.glob_file_list(self.LQ_root)
        self.subfolders_GT = sorted(glob.glob(self.GT_root+'/*'))
        
                            

    def __getitem__(self, index):  ## The index indicates the frame index which is flattened
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        GT_size = 256   # original = 512
        folder_path = self.subfolders_GT[index]
        img_paths_GT = sorted(glob.glob(folder_path+'/*mat'))[0:self.opt['N_frames']]
        rggb_l = []
        noise_l = []
        for img_path in img_paths_GT:
            mat_data = scio.loadmat(img_path)
            rggb = mat_data['param']['rggb'][0,0]
            noise = mat_data['param']['noise'][0,0]
            wb = mat_data['param']['lum'][0,0]
            noise_l.append(noise)
            rggb_l.append(rggb)
        
        wb = 1.0/wb
        rggb_case = np.stack(rggb_l,0)    
        rggb_in = np.transpose(rggb_case,(0,3,1,2)) # N,C,H,W
         
        noise = np.stack(noise_l,2) 
        #print(noise.shape)
        noise = noise[0,:,0]
        noise_p = (noise[0] + noise[2] + noise[4])/3.0
        noise_r = (noise[1] + noise[3] + noise[5])/3.0
        noise_map = (noise_p + noise_r)**0.5
        noise_map = np.tile(noise_map.reshape(1,1,1,1),[1,1,GT_size,GT_size])
        #LQ_size_tuple = (4,512,512)
        rggb_in = rggb_in[:,:,0:GT_size, 0:GT_size]
        noise_in_unpro_s = np.stack([noise[0],noise[2],noise[2],noise[4]],0).reshape(1,4,1,1)
        noise_in_unpro_r = np.stack([noise[1],noise[3],noise[3],noise[5]],0).reshape(1,4,1,1)
        noise_in_unpro = rggb_in*noise_in_unpro_s + noise_in_unpro_r
        noise_in_unpro = torch.from_numpy(np.ascontiguousarray(noise_in_unpro)).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(rggb_in)).float()
        noise_in = torch.from_numpy(np.ascontiguousarray(noise_map)).float()

        return {'LQs':img_LQs,'noise_in':noise_in,'folder':folder_path,'noise_in_unpro':noise_in_unpro,'wb':wb}

    def __len__(self):
        return len(self.subfolders_GT)
    
class Real_static(data.Dataset):


    def __init__(self, opt):
        super(Real_static, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
        # GS: No need LR
        #self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': []}   ## They are flattened
        #### Generate data info and cache data
        
        # subfolders_LQ = util.glob_file_list(self.LQ_root)
        self.subfolders_GT = sorted(glob.glob(self.GT_root+'/*'))
        
                            

    def __getitem__(self, index):  ## The index indicates the frame index which is flattened
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        GT_size = 384   # original = 512
        folder_path = self.subfolders_GT[index]
        img_paths = sorted(glob.glob(folder_path+'/0*mat'))[0:self.opt['N_frames']]
        gt_path = folder_path+'/gt.mat'
        rggb_l = []
        noise_l = []
        for img_path in img_paths:
            mat_data = scio.loadmat(img_path)
            rggb = mat_data['param']['rggb'][0,0]
            noise = mat_data['param']['noise'][0,0]
            wb = mat_data['param']['lum'][0,0]
            noise_l.append(noise)
            if self.opt['pre_demosaic'] == True:
                rggb = util.demosaic(rggb)
            rggb_l.append(rggb)
            
        wb = 1.0/wb 
        rggb_case = np.stack(rggb_l,0)    
        rggb_in = np.transpose(rggb_case,(0,3,1,2)) # N,C,H,W
        
        gt_data = scio.loadmat(gt_path)
        gt = gt_data['gt']
        gt = np.transpose(gt,(2,0,1))
         
        noise = np.stack(noise_l,2) 
        
        noise = noise[0,:,0]
        noise_p = (noise[0] + noise[2] + noise[4])/3.0
        noise_r = (noise[1] + noise[3] + noise[5])/3.0
        noise_map = (noise_p + noise_r)**0.5
        noise_map = np.tile(noise_map.reshape(1,1,1,1),[1,1,GT_size,GT_size]) if self.opt['pre_demosaic'] == False else np.tile(noise_map.reshape(1,1,1,1),[1,1,GT_size*2,GT_size*2])
        #LQ_size_tuple = (4,512,512)
        rggb_in = rggb_in[:,:,0:GT_size, 0:GT_size] if self.opt['pre_demosaic'] == False else rggb_in[:,:,0:GT_size*2, 0:GT_size*2]
        if self.opt['pre_demosaic'] == False:
            noise_in_unpro_s = np.stack([noise[0],noise[2],noise[2],noise[4]],0).reshape(1,4,1,1)
            noise_in_unpro_r = np.stack([noise[1],noise[3],noise[3],noise[5]],0).reshape(1,4,1,1)
        else:
            noise_in_unpro_s = np.stack([noise[0],noise[2],noise[4]],0).reshape(1,3,1,1)
            noise_in_unpro_r = np.stack([noise[1],noise[3],noise[5]],0).reshape(1,3,1,1)
        noise_in_unpro = rggb_in*noise_in_unpro_s + noise_in_unpro_r
        gt = gt[:,0:GT_size*2,0:GT_size*2]
         
        img_LQs = torch.from_numpy(np.ascontiguousarray(rggb_in)).float()
        noise_in = torch.from_numpy(np.ascontiguousarray(noise_map)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).float()
        noise_in_unpro = torch.from_numpy(np.ascontiguousarray(noise_in_unpro)).float()

        return {'LQs':img_LQs,'noise_in':noise_in,'GT':gt,'folder':folder_path,'noise_in_unpro':noise_in_unpro,'wb':wb}

    def __len__(self):
        return len(self.subfolders_GT)

