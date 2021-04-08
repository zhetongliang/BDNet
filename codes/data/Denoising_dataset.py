'''
REDS dataset
support reading images from lmdb, image folder and memcached

GuoShi: rewrite it for multi frame denoising
'''
import os.path as osp
import random

import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')

class Burst_static_dataset(data.Dataset):
    def __init__(self, opt):
        super(Burst_static_dataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.N_frames = opt['N_frames']
        #self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_type = self.opt['data_type'] # Where is the data_type come from ???
        # self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        #### directly load image keys
        self.paths_GT = util._get_paths_from_lmdb_paper(opt['dataroot_GT'])
        logger.info('Using lmdb meta info for cache keys.')

        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            #self.GT_env, self.LQ_env = None, None
            self.GT_env = None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

        # init noise parameters
        self.gain = 0
        self.sigma = 0
        self.gamma = 2.2

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        '''
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        '''

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img
    def random_shift_bdn(self,gt,rggb_in,ps=128,small_range=4,large_range=16): 
        '''
        rggb_in = np.transpose(rggb_in,(0,2,3,1)) 
        rggb_in = [util.demosaic(rggb_in[i,:,:,:]) for i in range(self.N_frames)]
        rggb_in = np.stack(rggb_in,0)   
        rggb_in = np.transpose(rggb_in,(0,3,1,2))
        '''
        N,C,H,W = rggb_in.shape[0],rggb_in.shape[1],rggb_in.shape[2],rggb_in.shape[3]
        rnd_h = random.randint(large_range, max(0, H - ps - large_range))
        rnd_w = random.randint(large_range, max(0, W - ps - large_range))
        gt = gt[:,rnd_h*2:rnd_h*2+ps*2,rnd_w*2:rnd_w*2+ps*2]


        x_small= rggb_in[:,:,rnd_h-small_range:rnd_h+ps+small_range,rnd_w-small_range:rnd_w+ps+small_range]
        N_s,C_s,H_s,W_s = x_small.shape[0],x_small.shape[1],x_small.shape[2],x_small.shape[3]
        x_large = rggb_in[:,:,rnd_h-large_range:rnd_h+ps+large_range,rnd_w-large_range:rnd_w+ps+large_range]
        N_l,C_l,H_l,W_l = x_large.shape[0],x_large.shape[1],x_large.shape[2],x_large.shape[3]
        new_l = []
        for i in range(N):
            if i != N//2:
                flag = random.randint(0, 8)
                if flag>1:
                    rnd_h_ = random.randint(0, max(0, H_l-ps))
                    rnd_w_ = random.randint(0, max(0, W_l-ps))
                    x_cur = x_large[i,:,rnd_h_:rnd_h_+ps,rnd_w_:rnd_w_+ps]
                else:
                    rnd_h_ = random.randint(0, max(0, H_s-ps))
                    rnd_w_ = random.randint(0, max(0, W_s-ps))
                    x_cur = x_small[i,:,rnd_h_:rnd_h_+ps,rnd_w_:rnd_w_+ps]
                new_l.append(x_cur)
            else:
                x_cur = rggb_in[i,:,rnd_h:rnd_h+ps,rnd_w:rnd_w+ps]
                new_l.append(x_cur)

        x_new = np.stack(new_l,0)
        return gt,x_new
    def random_shift_kpn(self,gt,rggb_in,ps,small_range=2,large_range=16): 
        rggb_in = np.transpose(rggb_in,(0,2,3,1)) 
        rggb_in = [util.demosaic(rggb_in[i,:,:,:]) for i in range(self.N_frames)]
        rggb_in = np.stack(rggb_in,0)   
        rggb_in = np.transpose(rggb_in,(0,3,1,2))
        N,C,H,W = rggb_in.shape[0],rggb_in.shape[1],rggb_in.shape[2],rggb_in.shape[3]
        rnd_h = random.randint(large_range, max(0, H - ps - large_range))
        rnd_w = random.randint(large_range, max(0, W - ps - large_range))
        gt = gt[:,rnd_h:rnd_h+ps,rnd_w:rnd_w+ps]


        x_small= rggb_in[:,:,rnd_h-small_range:rnd_h+ps+small_range,rnd_w-small_range:rnd_w+ps+small_range]
        N_s,C_s,H_s,W_s = x_small.shape[0],x_small.shape[1],x_small.shape[2],x_small.shape[3]
        x_large = rggb_in[:,:,rnd_h-large_range:rnd_h+ps+large_range,rnd_w-large_range:rnd_w+ps+large_range]
        N_l,C_l,H_l,W_l = x_large.shape[0],x_large.shape[1],x_large.shape[2],x_large.shape[3]
        new_l = []
        for i in range(N):
            if i != N//2:
                flag = random.randint(0, 8)
                if flag>1:
                    rnd_h_ = random.randint(0, max(0, H_l-ps))
                    rnd_w_ = random.randint(0, max(0, W_l-ps))
                    x_cur = x_large[i,:,rnd_h_:rnd_h_+ps,rnd_w_:rnd_w_+ps]
                else:
                    rnd_h_ = random.randint(0, max(0, H_s-ps))
                    rnd_w_ = random.randint(0, max(0, W_s-ps))
                    x_cur = x_small[i,:,rnd_h_:rnd_h_+ps,rnd_w_:rnd_w_+ps]
                new_l.append(x_cur)
            else:
                x_cur = rggb_in[i,:,rnd_h:rnd_h+ps,rnd_w:rnd_w+ps]
                new_l.append(x_cur)

        x_new = np.stack(new_l,0)
        return gt,x_new
            
    def __getitem__(self, index):
        index = int(index)
        if self.data_type == 'mc':
            self._ensure_memcached()
        #elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
        elif self.data_type == 'lmdb' and self.GT_env is None:
            self._init_lmdb()

        # scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        rggb_in,noise,rgb_gt = util.read_img_paper_real(self.GT_env, key)
        
   
        ##### Process noise #####
        noise = noise[0,:]
        noise_p = (noise[0] + noise[2] + noise[4])/3.0
        noise_r = (noise[1] + noise[3] + noise[5])/3.0
        noise_map = (noise_p + noise_r)**0.5
        noise_map = np.tile(noise_map.reshape(1,1,1,1),[1,1,GT_size*2,GT_size*2])
        
        ##### Process input and output ######
        rggb_in = np.transpose(rggb_in,(0,3,1,2))
        rgb_gt = np.transpose(rgb_gt,(2,0,1))
        
        is_kpn = True
        LQ_size_tuple = (4,256,256)  # rggb of SIDD iPhone is 256, rgb is 512
        #if self.opt['phase'] == 'train1':
        C, H, W = LQ_size_tuple  # LQ size
        # randomly crop
        # GS: No LR_input input
        rnd_h = random.randint(0, max(0, H - GT_size))
        rnd_w = random.randint(0, max(0, W - GT_size))
        
        if self.opt['pre_demosaic'] is False:
            rgb_gt,rggb_in = self.random_shift_bdn(rgb_gt,rggb_in,ps=GT_size,small_range=4,large_range=12)
        else:
            rgb_gt,rggb_in = self.random_shift_kpn(rgb_gt,rggb_in,ps=GT_size,small_range=4,large_range=16)                
        # augmentation - flip, rotate
        rggb_in,rgb_gt = util.augment_paper_real(rggb_in,rgb_gt, self.opt['use_flip'], self.opt['use_rot'])

        img_GT = torch.from_numpy(np.ascontiguousarray(rgb_gt)).float()  ## You can see that there are only 3 dims for an image
        img_LQs = torch.from_numpy(np.ascontiguousarray(rggb_in)).float()
        noise_in = torch.from_numpy(np.ascontiguousarray(noise_map)).float()

        return {'LQs': img_LQs, 'GT': img_GT,'key':key,'ind':index,'noise_in':noise_in}

    def __len__(self):
        return len(self.paths_GT)


class Synthetic_dynamic_dataset(data.Dataset):
    def __init__(self, opt):
        super(Synthetic_dynamic_dataset, self).__init__()
        self.opt = opt
        self.noise_type = opt['noise_type']
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        #self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_type = self.opt['data_type'] # Where is the data_type come from ???
        # self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        #### directly load image keys
        self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        logger.info('Using lmdb meta info for cache keys.')

        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            #self.GT_env, self.LQ_env = None, None
            self.GT_env = None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

        # init noise parameters
        self.gain = 0
        self.sigma = 0
        self.gamma = 2.2

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        '''
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        '''

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        index = int(index*5)
        if self.data_type == 'mc':
            self._ensure_memcached()
        #elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
        elif self.data_type == 'lmdb' and self.GT_env is None:
            self._init_lmdb()

        # scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        # name_a: folder, name_b: image name
        name_a, _ = key.split('_')   ### 000000_1
        # reference image
        center_frame_idx = 3

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)

        # get the neighbor list
        neighbor_list = list(  #### mainly name_b   start from 1 not 0
            range(center_frame_idx - self.half_N_frames * interval,
                  center_frame_idx + self.half_N_frames * interval + 1, interval))

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))
        img_size_H,img_size_W = 256,448
        LQ_size_tuple = (3, img_size_H, img_size_W)

        #### get the GT image (as the center frame)
        # GS: running this loop
        #img_GT = util.read_img(self.GT_env, key, (3, 720, 1280))
        #### Init noise parameters
        red_gain  =  1.9 + np.random.rand(1)*(2.4-1.9)  
        red_gain = red_gain.reshape(1,1,1)
        blue_gain =  1.5 + np.random.rand(1)*(1.9-1.5) 
        blue_gain = blue_gain.reshape(1,1,1)
        wb_gain = 1.0/np.concatenate([red_gain,np.ones_like(red_gain,dtype=np.float32),blue_gain],2)
        if self.noise_type == "Poisson_Gaussian":
            max_gain = 0.01
            min_gain = 0.00001
            self.gain = min_gain + np.random.rand(3).reshape(1,1,3) * (max_gain - min_gain)
            self.sigma = np.random.rand(3).reshape(1,1,3) * (15.0/255.0)
            if self.opt['noise_level'] == "large":
                self.gain = min_gain*100.0 + np.random.rand(3).reshape(1,1,3) * (max_gain - min_gain*100.0)
                self.sigma = np.random.rand(3).reshape(1,1,3) * (30.0/255.0) + (10.0/255.0)
            self.noise_in = (self.gain + self.sigma**2.0)**0.5
            self.noise_in = np.mean(self.noise_in,axis=2,keepdims=True)
            self.noise_in = np.tile(self.noise_in,[img_size_H,img_size_W,1])
            self.noise_in = np.expand_dims(np.transpose(self.noise_in,[2,0,1]),0)
        elif self.noise_type == "Fixed_Gaussian":           
            self.sigma = np.float(int(self.opt['sigma']))/255.0
            self.sigma = np.array(self.sigma).reshape(1,1,1)
            self.sigma = np.tile(self.sigma,[1,1,3])
            self.gain = None
            self.noise_in = np.mean(self.sigma,axis=2,keepdims=True)
            self.noise_in = np.tile(self.noise_in,[img_size_H,img_size_W,1])
            self.noise_in = np.expand_dims(np.transpose(self.noise_in,[2,0,1]),0)

        self.gamma = 2.0 + np.random.rand(1) * 0.5
        
        img_GT = util.read_img(self.GT_env, '{}_{}'.format(name_a, center_frame_idx), (3, img_size_H, img_size_W))
        img_GT = util.func_degamma(img_GT, self.gamma) * wb_gain
        if self.opt['GT_noise']==True:
            img_GT = util.add_noise_c(img_GT, self.gain, self.sigma)

        img_LQ_l = []
        for v in neighbor_list:
            # img_LQ_path = osp.join(self.LQ_root, name_a, '{:08d}.png'.format(v))
                # GS: running this loop
                # img_LQ = util.read_img(self.LQ_env, '{}_{:08d}'.format(name_a, v), LQ_size_tuple)
            img_LQ = util.read_img(self.GT_env, '{}_{}'.format(name_a, v), LQ_size_tuple)
            #img_LQ = util.RGB2Gray(img_LQ)
            img_LQ = util.func_degamma(img_LQ, self.gamma) * wb_gain
            img_LQ = util.mosaic(img_LQ)
            img_LQ = util.add_noise_m(img_LQ, self.gain, self.sigma,self.noise_type)
            if self.opt['pre_demosaic'] is True:
                img_LQ = util.demosaic(img_LQ)

            img_LQ_l.append(img_LQ)
        

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            # GS: No LR_input input
            rnd_h = random.randint(0, max(0, H//2 - GT_size//2))
            rnd_w = random.randint(0, max(0, W//2 - GT_size//2))
            if self.opt['pre_demosaic'] is False:
                img_LQ_l = [v[rnd_h:rnd_h + GT_size//2, rnd_w:rnd_w + GT_size//2, :] for v in img_LQ_l]
            else:
                img_LQ_l = [v[rnd_h*2:rnd_h*2 + GT_size, rnd_w*2:rnd_w*2 + GT_size, :] for v in img_LQ_l]
            img_GT = img_GT[rnd_h*2:rnd_h*2 + GT_size, rnd_w*2:rnd_w*2 + GT_size, :]
            
            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_GT = img_GT[:, :, [2, 1, 0]]
        # img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()  ## You can see that there are only 3 dims for an image
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        noise_in = torch.from_numpy(np.ascontiguousarray(self.noise_in)).float()
        
        return {'LQs': img_LQs, 'GT': img_GT, 'key':'{}_{}'.format(name_a, center_frame_idx), 'ind':index,'noise_in':noise_in}

    def __len__(self):
        return len(self.paths_GT)//5

    
