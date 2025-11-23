import utils.utils as utils


def create_parser():
    import configargparse
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
    p.add_argument('--data_path', type=str, default='data/pics', help='Specify data path: dataset or split-lohmann')
    p.add_argument('--device', type=str, default='cuda', help='Choose device [cpu|cuda]')
    p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
    p.add_argument('--prop_dist', type=int, default=200, help='Propagation distance in mm')
    p.add_argument('--output_dir', type=str, default='results/', help='Output folder por phase patterns and results')
    p.add_argument('--proptype', type=str, default='ASM', help='Ideal propagation model')
    p.add_argument('--generator_path', type=str, help='Torch save of Holonet, start from pre-trained gen. Leave as "" to train from scratch' \
                   'For inference, set several paths separated by "," to enable time multiplexing')
    p.add_argument('--batch_size', type=int, default=4, help='Size of minibatch')
    p.add_argument('--scale_output', type=float, default=0.95,
                help='Scale of output applied to reconstructed intensity from SLM')
    p.add_argument('--perfect_prop_model', type=utils.str2bool, default=True,
                help='Use ideal ASM as the loss function')
    p.add_argument('--manual_aberr_corr', type=utils.str2bool, default=True,
                help='Divide source amplitude manually, (possibly apply inverse zernike of primal domain')
    p.add_argument('--set_dc_block', type=utils.str2bool, default=False, help="Filters out the hologram's dc term")
    return p


def add_train_args(p):
    p.add_argument('--run_id', type=str, default='', help='Experiment name', required=True)
    p.add_argument('--use_loss_mixture', type=utils.str2bool, default=True, help='do')
    p.add_argument('--flag_randominit', type=utils.str2bool, default=False, help='do')
    p.add_argument('--purely_unet', type=utils.str2bool, default=False, help='Use U-Net for entire estimation if True')
    p.add_argument('--disable_loss_amp', type=utils.str2bool, default=True, help='Disable manual amplitude loss')
    p.add_argument('--num_latent_codes', type=int, default=2, help='Number of latent codes of trained prop model.')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate of Holonet weights')
    p.add_argument('--step_lr', type=utils.str2bool, default=False, help='Use of lr scheduler')
    p.add_argument('--use_mse_init', type=utils.str2bool, default=False, help='First 500  will be MSE regardless of loss_fun')
    p.add_argument('--lambda_uniform', type=float, default=1.0, help='Hyperparameter for Uniform Distance loss')
    p.add_argument('--lambda_rayleigh', type=float, default=1.0, help='Hyperparameter for Rayleigh Distance loss')
    p.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    p.add_argument('--prop_dist_offset', type=float, default=0, help='Propagation distance offset in mm (3D)')
    p.add_argument('--model_path', type=str, default='', help='Torch save CITL-calibrated model')


def add_inference_args(p):
    p.add_argument('--use_fp16', action='store_true', help='Autocast network to float16 for performance')
    p.add_argument('--quantize', action='store_true', help='Quantize SLM phase pattern')
    p.add_argument('--prop_dist_offset', type=float, default=5, help='Propagation distance offset in mm (3D)')
    p.add_argument('--set_prepupil_obstructions', action='store_true', help='Set pre-pupil obstructions')
    p.add_argument('--set_postpupil_obstructions', action='store_true', help='Set post-pupil obstructions')
    # p.add_argument('--model_lut', type=utils.str2bool, default=True, help='Activate the lut of model')
    