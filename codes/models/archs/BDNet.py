''' network architecture for EDVR '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import models.archs.arch_util as arch_util
try:
    from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
            
    
class Predenoise_Pyramid(nn.Module):
    def __init__(self, nf=64,num_in=4,num_out=64,out_act='lrelu'):
        '''
        HR_in: True if the inputs are high spatial size
        '''
        super(Predenoise_Pyramid, self).__init__()
        
        self.conv_first = nn.Conv2d(num_in, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ConvBlock_noBN, nf=nf)
        self.dn_L10_conv = basic_block()
        self.dn_L11_conv = basic_block()
        self.down_L1_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.dn_L20_conv = basic_block()
        self.dn_L21_conv = basic_block()
        self.down_L2_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.dn_L30_conv = basic_block()
        self.dn_L31_conv = basic_block()
        self.dn_L20_conv_up = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.dn_L21_conv_up = basic_block()
        self.dn_L10_conv_up = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.dn_L11_conv_up = basic_block()       
        self.last_conv = nn.Conv2d(nf, num_out, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.out_act = out_act

    def forward(self, x):
        L1_fea = self.lrelu(self.conv_first(x))
        L1_fea = self.lrelu(self.dn_L10_conv(L1_fea))
        L1_fea = self.lrelu(self.dn_L11_conv(L1_fea))
        L2_fea = self.lrelu(self.down_L1_conv(L1_fea))
        
        L2_fea = self.lrelu(self.dn_L20_conv(L2_fea))
        L2_fea = self.lrelu(self.dn_L21_conv(L2_fea))        
        L3_fea = self.lrelu(self.down_L2_conv(L2_fea))
        
        L3_fea = self.lrelu(self.dn_L30_conv(L3_fea))
        L3_fea = self.lrelu(self.dn_L31_conv(L3_fea))        
        L2_fea_up = F.interpolate(L3_fea, scale_factor=2, mode='bilinear',
                               align_corners=False)
        
        L2_fea_up = torch.cat([L2_fea_up, L2_fea], dim=1)
        L2_fea_up = self.lrelu(self.dn_L20_conv_up(L2_fea_up)) 
        L2_fea_up = self.lrelu(self.dn_L21_conv_up(L2_fea_up))   
        L1_fea_up = F.interpolate(L2_fea_up, scale_factor=2, mode='bilinear',
                               align_corners=False)
        
        L1_fea_up = torch.cat([L1_fea_up, L1_fea], dim=1)
        L1_fea_up = self.lrelu(self.dn_L10_conv_up(L1_fea_up)) 
        L1_fea_up = self.lrelu(self.dn_L11_conv_up(L1_fea_up))  
        out = self.last_conv(L1_fea_up)
        if self.out_act == 'lrelu':
            out = self.lrelu(out)  
        elif self.out_act == 'sigmoid':
            out = self.sigmoid(out)
            
        return out
    
    
class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))   # changed
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))   # changed
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))

        return L1_fea

class Fusion(nn.Module):

    def __init__(self, nf=64, nframes=5, center=2):
        super(Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob
        fea = self.lrelu(self.fea_fusion(aligned_fea))
        return fea


class BDNet_train(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,is_noise=True):
        super(BDNet_train, self).__init__()
        self.nf = nf
        self.is_noise = is_noise
        self.center = nframes // 2 if center is None else center
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        
        ## PreP modules, one for synthetic noise and the other for real noise
        self.pre_dn_syn = Predenoise_Pyramid(nf=nf,num_in=(5 if is_noise == True else 4),num_out=nf)
        self.pre_dn_real = Predenoise_Pyramid(nf=nf,num_in=(5 if is_noise == True else 4),num_out=nf)
        self.PreP_out = nn.Conv2d(nf, 12, 1, 1, 0, bias=True)
        
        ## TemP module
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = Fusion(nf=nf, nframes=nframes, center=self.center)
        self.TemP_out = nn.Conv2d(nf, 12, 1, 1, 0, bias=True)

        #### PostP modules, one for synthetic noise and the other for real noise
        self.recon_trunk_syn = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        self.HRconv_syn = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last_syn = nn.Conv2d(64, 12, 3, 1, 1, bias=True)
        
        self.recon_trunk_real = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        self.HRconv_real = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last_real = nn.Conv2d(64, 12, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x,noise,flag='syn'):
        B, N, C, H, W = x.size()  # N video frames
        #x_center = x[:, self.center, :, :, :].contiguous()
        # L1
        if self.is_noise == True:
            noise = noise[:,:,:,0:H,0:W]
            noise = noise.repeat(1,N,1,1,1)
            noise = noise.reshape(-1,1,H,W)
            input = torch.cat([x.view(-1, C, H, W),noise],1)
        else:
            input = x.view(-1, C, H, W)
            
        ## PreP module
        if flag == 'syn':
            L1_fea = self.pre_dn_syn(input)
            L1_fea_ref = L1_fea.reshape(B,N,self.nf, H, W)
            L1_fea_ref = L1_fea_ref[:,self.center,:,:,:]
            pre_out = F.pixel_shuffle(self.PreP_out(L1_fea_ref),2)
        else:
            L1_fea = self.pre_dn_real(input)
            L1_fea_ref = L1_fea.reshape(B,N,self.nf, H, W)
            L1_fea_ref = L1_fea_ref[:,self.center,:,:,:]
            pre_out = F.pixel_shuffle(self.PreP_out(L1_fea_ref),2)
        
        ## TemP module
        L1_fea = self.feature_extraction(L1_fea)
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        fea = self.fusion(aligned_fea)
        align_out = F.pixel_shuffle(self.TemP_out(fea),2)

        ## PostP
        if flag == 'syn':
            out = F.pixel_shuffle(self.conv_last_syn(self.lrelu(self.HRconv_syn(self.recon_trunk_syn(fea)))),2)
        else:
            out = F.pixel_shuffle(self.conv_last_real(self.lrelu(self.HRconv_real(self.recon_trunk_real(fea)))),2)
        return pre_out,align_out,out
    
    
class BDNet_test(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,is_noise=True):
        super(BDNet_test, self).__init__()
        self.nf = nf
        self.is_noise = is_noise
        self.center = nframes // 2 if center is None else center
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        
        ## PreP module
        self.pre_dn_real = Predenoise_Pyramid(nf=nf,num_in=(5 if is_noise == True else 4),num_out=nf)
        
        ## TemP module
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = Fusion(nf=nf, nframes=nframes, center=self.center)

        #### PostP module
        self.recon_trunk_real = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        self.HRconv_real = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last_real = nn.Conv2d(64, 12, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x,noise):
        B, N, C, H, W = x.size()  # N video frames
        #x_center = x[:, self.center, :, :, :].contiguous()
        # L1
        if self.is_noise == True:
            noise = noise[:,:,:,0:H,0:W]
            noise = noise.repeat(1,N,1,1,1)
            noise = noise.reshape(-1,1,H,W)
            input = torch.cat([x.view(-1, C, H, W),noise],1)
        else:
            input = x.view(-1, C, H, W)
            
        ## PreP module
        L1_fea = self.pre_dn_real(input)
        L1_fea_ref = L1_fea.reshape(B,N,self.nf, H, W)
        L1_fea_ref = L1_fea_ref[:,self.center,:,:,:]
        
        ## TemP module
        L1_fea = self.feature_extraction(L1_fea)
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        fea = self.fusion(aligned_fea)

        ## PostP module
        out = F.pixel_shuffle(self.conv_last_real(self.lrelu(self.HRconv_real(self.recon_trunk_real(fea)))),2)
        return out   