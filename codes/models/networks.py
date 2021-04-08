import torch
import models.archs.BDNet as BDNet

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration

    if which_model == 'BDNet':
        netG = BDNet.BDNet_train(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'])
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(which_model))
    return netG


