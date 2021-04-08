import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def bayer_mask_tensor(rgb,cfa):
    if cfa == 'bggr':
        r = rgb[:,0:1,1::2,1::2]
        g1 = rgb[:,1:2,0::2,1::2]
        g2 = rgb[:,1:2,1::2,0::2]
        b = rgb[:,2:3,0::2,0::2]
    elif cfa == 'rggb':
        r = rgb[:,0:1,0::2,0::2]
        g1 = rgb[:,1:2,0::2,1::2]
        g2 = rgb[:,1:2,1::2,0::2]
        b = rgb[:,2:3,1::2,1::2]
    elif cfa == 'grbg':
        r = rgb[:,0:1,0::2,1::2]
        g1 = rgb[:,1:2,0::2,0::2]
        g2 = rgb[:,1:2,1::2,1::2]
        b = rgb[:,2:3,1::2,0::2]
    elif cfa == 'gbrg':
        r = rgb[:,0:1,1::2,0::2]
        g1 = rgb[:,1:2,0::2,0::2]
        g2 = rgb[:,1:2,1::2,1::2]
        b = rgb[:,2:3,0::2,1::2]

    rggb = torch.cat([r,g1,g2,b],1)
    return rggb
        

def fea_warp(fea,offset):
    
    B,C,H,W = fea.size()   # B,C,H,W
    
    offset_p = offset.permute(0,2,3,1)  # B,H,W,2
    grid = torch.meshgrid(torch.linspace(0.0,H-1,H), torch.linspace(0.0,W-1,W))   # 0~1 ij
    grid = torch.reshape(torch.stack(grid, axis=-1),[1,H,W,2]).cuda()  ## 1,H,W,2

    offset = offset_p + grid  # B,H,W,2
    i_idx,j_idx = offset[:,:,:,0:1],  offset[:,:,:,1:2]
    #i_idx,j_idx = i_idx*(H-1.0), j_idx*(W-1.0)
    i_idx,j_idx = torch.clamp(i_idx,0.0,H-1),torch.clamp(j_idx,0.0,W-1)
    i_idx0,j_idx0 = torch.floor(i_idx), torch.floor(j_idx)
    i_idx1,j_idx1 = i_idx0 + 1, j_idx0 + 1
    #i_idx0,j_idx0 = torch.clamp(i_idx0,0.0,H).long(),torch.clamp(j_idx0,0.0,W).long()
    #i_idx1,j_idx1 = torch.clamp(i_idx1,0.0,H).long(),torch.clamp(j_idx1,0.0,W).long() # B,H,W,1
    
    b_idx = torch.arange(0, B).reshape(B,1,1,1).repeat(1,H,W,1).long().cuda()
    def _get_value(input,b_idx,i_idx,j_idx):
        '''
        input: B,H,W,C
        i,j: B,H,W,1
        '''
        B,H,W,C = b_idx.size(0),b_idx.size(1),b_idx.size(2),input.size(3)
        b_idx,i_idx,j_idx = b_idx.reshape(-1),i_idx.reshape(-1),j_idx.reshape(-1)
        input_f = input.reshape(-1,C)
        ind = b_idx*(H+1)*(W+1) + i_idx*(W+1) + j_idx
        input_n = input_f.index_select(0,ind)
        input_n = input_n.reshape(B,H,W,C)
        return input_n
    
    fea_p = F.pad(fea,[0,1,0,1],'replicate') 
    fea_p = fea_p.permute(0,2,3,1)  # B,H,W,C
    fea00 = _get_value(fea_p,b_idx,i_idx0.long(),j_idx0.long()) 
    fea01 = _get_value(fea_p,b_idx,i_idx0.long(),j_idx1.long()) 
    fea10 = _get_value(fea_p,b_idx,i_idx1.long(),j_idx0.long()) 
    fea11 = _get_value(fea_p,b_idx,i_idx1.long(),j_idx1.long()) 
    
    i_idx0,j_idx0,i_idx1,j_idx1 = i_idx0.float(),j_idx0.float(),i_idx1.float(),j_idx1.float()  
    w00 = torch.clamp_min(1.0-torch.abs(i_idx-i_idx0),0.0)*torch.clamp_min(1.0-torch.abs(j_idx-j_idx0),0.0)
    w01 = torch.clamp_min(1.0-torch.abs(i_idx-i_idx0),0.0)*torch.clamp_min(1.0-torch.abs(j_idx-j_idx1),0.0)
    w10 = torch.clamp_min(1.0-torch.abs(i_idx-i_idx1),0.0)*torch.clamp_min(1.0-torch.abs(j_idx-j_idx0),0.0)
    w11 = torch.clamp_min(1.0-torch.abs(i_idx-i_idx1),0.0)*torch.clamp_min(1.0-torch.abs(j_idx-j_idx1),0.0)
    
    fea_out = fea00*w00 + fea01*w01 + fea10*w10 + fea11*w11
    fea_out = fea_out.permute(0,3,1,2)
    return fea_out

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity + out
    
class ConvBlock_noBN(nn.Module):
    '''Conv block w/o BN
    ---Conv-ReLU-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ConvBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1], 0.1)

    def forward(self, x):
        out = self.conv1(x)
        return out
    
class ConvBlocks(nn.Module):
    '''Conv block w/o BN
    ---Conv-ReLU-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ConvBlocks, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # initialization
        initialize_weights([self.conv1], 0.1)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        return out

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
