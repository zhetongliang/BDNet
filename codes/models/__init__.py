import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'sr':  # PSNR-oriented super resolution
        from .SR_model import SRModel as M
    elif model == 'srgan':  # GAN-based super resolution, SRGAN / ESRGAN
        from .SRGAN_model import SRGANModel as M
    # video restoration
    elif model == 'video_base':
        from .Video_base_model import VideoBaseModel as M
    elif model == 'video_base_rnn':
        from .Video_base_model import VideoBaseModel_RNN as M
    elif model == 'video_base_paper_ft':
        from .Video_base_model import VideoBaseModel_paper_finetune as M
    elif model == 'video_base_paper_m':
        from .Video_base_model import VideoBaseModel_modified_paper as M
    elif model == 'video_base_multi_unprocess':
        from .Video_base_model import VideoBaseModel_multi_unprocess as M
    elif model == 'video_base_multi_unprocess_vivo':
        from .Video_base_model import VideoBaseModel_multi_unprocess_vivo as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
