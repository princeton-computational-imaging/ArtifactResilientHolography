"""
Real-Time ARH

This is the main script used for training the Real-Time ARH which is inspired by HoloNet [Peng et al. 2020].

Usage
-----

$ python train_ARH.py --channel=1 --run_id=experiment_1

"""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
import utils.utils as utils
import utils.perceptualloss as perceptualloss
from propagation_model import ModelPropagate
from holonet import *
from utils.augmented_image_loader import ImageLoader

import holo_utils as hu
import parser
from constants import get_constants


def main(opt):
    # Command line argument processing
    warm_start = opt.generator_path is not None
    num_smooth_iters = 500
    num_smooth_epochs = 2 if not warm_start else 0
    opt.num_epochs += 1 if opt.use_mse_init and not warm_start else 0

    # Propagation parameters
    holo_params = get_constants(opt)

    chan_str = holo_params.chan_str
    print(holo_params)

    # tensorboard setup and file naming
    time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
    writer = SummaryWriter(f'runs/{opt.run_id}_{chan_str}_{time_str}')

    # Training parameters
    device = torch.device('cuda')
    data_path = 'data/DIV2K_train_HR'  # path for training data

    if opt.model_path == '':
        opt.model_path = f'./models/{chan_str}.pth'

    # resolutions need to be divisible by powers of 2 for unet


    ###############
    # Load models #
    ###############

    # set model instance as naive ASM
    model_prop = ModelPropagate(distance=holo_params.prop_dist, feature_size=holo_params.feature_size,
                                wavelength=holo_params.wavelength, target_field=False, num_gaussians=0,
                                num_coeffs_fourier=0, use_conv1d_mlp=False, num_latent_codes=[0],
                                norm=None, blur=None, content_field=False, proptype=opt.proptype).to(device)
    model_prop.eval()  # ensure freezing propagation model

    # create new phase generator object

    phase_generator = HoloNet(
        distance=holo_params.prop_dist,
        wavelength=holo_params.wavelength,
        feature_size=holo_params.feature_size,
        initial_phase=InitialPhaseUnet(4, 16),
        final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2),
        proptype=opt.proptype).to(device)

    if warm_start:
        phase_generator.load_state_dict(torch.load(opt.generator_path, map_location=device))

    phase_generator.train()  # generator to be trained


    ###################
    # Set up training #
    ###################

    optvars = phase_generator.parameters()

    # set aside the VGG loss for the first num_mse_epochs
    vgg_loss = perceptualloss.PerceptualLoss(lambda_feat=0.025).to(device)
    mse_loss = nn.MSELoss().to(device)
    loss = mse_loss

    # create optimizer
    if warm_start:
        opt.lr /= 10
    optimizer = optim.Adam(optvars, lr=opt.lr)

    # loads images from disk, set to augment with flipping
    image_loader = ImageLoader(data_path,
                            channel=opt.channel,
                            batch_size=opt.batch_size,
                            image_res=holo_params.image_res,
                            homography_res=holo_params.homography_res,
                            shuffle=True,
                            vertical_flips=True,
                            horizontal_flips=True)


    # learning rate scheduler
    if opt.step_lr:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 500, 0.5)


    #################
    # Training loop #
    #################

    print("epochs:", opt.num_epochs, "k:", len(image_loader))

    for i in range(opt.num_epochs):
        # switch to actual loss function from mse when desired
        if i == num_smooth_epochs or not opt.use_mse_init:
            print('Swapping from MSE to VGG')
            loss = vgg_loss

        for k, target in enumerate(image_loader):

            # get target image
            target_amp, target_res, target_filename = target
            target_amp = target_amp.to(device)

            optimizer.zero_grad()

            # forward model
            slm_amp, slm_phase = phase_generator(target_amp)
            output_complex = model_prop(slm_phase)
            output_complex = hu.process_dc_block(output_complex, opt.set_dc_block)
            output_lin_intensity = torch.sum(output_complex.abs()**2 * opt.scale_output, dim=1, keepdim=True)
            output_amp = torch.pow(output_lin_intensity, 0.5)

            # VGG assumes RGB input, we just replicate
            target_amp = target_amp.repeat(1, 3, 1, 1)
            output_amp = output_amp.repeat(1, 3, 1, 1)

            # crop outputs to the region we care about for loss
            output_amp = utils.crop_image(output_amp, holo_params.homography_res, stacked_complex=False)
            target_amp = utils.crop_image(target_amp, holo_params.homography_res, stacked_complex=False)

            # ignore the cropping and do full-image loss
            if 'nocrop' in opt.run_id:
                loss_nocrop = loss(output_amp, target_amp)

            # force equal mean amplitude when checking loss
            output_amp = output_amp * target_amp.mean(dim=(2,3), keepdim=True) / output_amp.mean(dim=(2,3), keepdim=True)

            # loss and optimize
            loss_main = loss(output_amp, target_amp)
            uniform_distance_loss = hu.uniform_distance_loss(slm_phase) * .45
            rayleigh_distance_loss = hu.rayleigh_distance_loss(output_complex, sigma = .1) * .6
            
            if warm_start or opt.disable_loss_amp:
                flag = 'No amplitude'
                loss_amp = 0
            else:
                # extra loss term to encourage uniform amplitude at SLM plane
                # helps some networks converge properly initially
                loss_amp = mse_loss(slm_amp.mean(), slm_amp)
                flag = 'Amplitude'

            loss_val = ((loss_nocrop if 'nocrop' in opt.run_id else loss_main)
                        + 0.1 * loss_amp)
            
            if opt.use_loss_mixture and i >= num_smooth_epochs:
                flag += ' + regularizers'
                loss_val = hu.loss_mix([loss_main, uniform_distance_loss, rayleigh_distance_loss],
                                        [1.0, opt.lambda_uniform, opt.lambda_rayleigh])

            loss_val.backward()
            optimizer.step()

            if opt.step_lr:
                scheduler.step()

            # print and output to tensorboard
            ik = k + i * len(image_loader)
            if opt.use_mse_init and i >= num_smooth_iters:
                ik += num_smooth_iters - len(image_loader)
            print(f'Training iteration {ik}: {loss_val.item()}, {uniform_distance_loss.item()}, {rayleigh_distance_loss.item()}, {flag}')

            with torch.no_grad():
                writer.add_scalar('Loss', loss_val, ik)

                if ik % 50 == 0:
                    # write images and loss to tensorboard
                    writer.add_image('Target Amplitude', target_amp[0, ...], ik)

                    # normalize reconstructed amplitude
                    output_amp0 = output_amp[0, ...]
                    maxVal = torch.max(output_amp0)
                    minVal = torch.min(output_amp0)
                    tmp = (output_amp0 - minVal) / (maxVal - minVal)
                    writer.add_image('Reconstruction Amplitude', tmp, ik)
                    eye = torch.fft.fftshift(torch.fft.fft2(output_complex[0, ...])).abs().repeat(3,1,1)
                    eye = eye/eye.max()
                    writer.add_image('Eyebox',eye, ik)

                    # normalize SLM phase
                    writer.add_image('SLM Phase', (slm_phase[0, ...] + math.pi) / (2 * math.pi), ik)

        # save trained model
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        print('Saving models')
        torch.save(phase_generator.state_dict(),
                f'checkpoints/{opt.run_id}_{chan_str}_{time_str}_{i+1}.pth')


if __name__ == '__main__':
    p = parser.create_parser()
    parser.add_train_args(p)
    opt = p.parse_args()
    main(opt)
