"""
This is the script that is used for the implementing our 3D realtime of Real-Time ARH. The class is adapted from HoloNet [Peng et al. 2020].

"""

import math
import numpy as np
import torch
import torch.nn as nn


import utils.utils as utils
from propagation_ASM import propagation_ASM, compute_zernike_basis, combine_zernike_basis
from utils.pytorch_prototyping.pytorch_prototyping import Conv2dSame, Unet
import matplotlib.pyplot as plt


class RealTimeARH3D(nn.Module):

    def __init__(self, distance=0.1, offset=0.001, wavelength=520e-9, feature_size=6.4e-6,
                 zernike_coeffs=None, source_amplitude=None, target_field=None, latent_codes=None,
                 initial_phase=None, final_phase_only=None, proptype='ASM', linear_conv=True,
                 manual_aberr_corr=False, phase_shift = 0):
        super(RealTimeARH3D, self).__init__()

        # submodules
        self.source_amplitude = source_amplitude
        self.initial_phase = initial_phase
        self.final_phase_only = final_phase_only
        if target_field is not None:
            self.target_field = target_field.detach()
        else:
            self.target_field = None

        if latent_codes is not None:
            self.latent_codes = latent_codes.detach()
        else:
            self.latent_codes = None

        # propagation parameters
        self.wavelength = wavelength
        self.feature_size = (feature_size
                             if hasattr(feature_size, '__len__')
                             else [feature_size] * 2)
        self.distance = -distance
        self.offset = -offset

        self.zernike_coeffs = (None if zernike_coeffs is None
                               else -zernike_coeffs.clone().detach())

        # objects to precompute
        self.zernike = None
        self.precomped_H_0 = None
        self.precomped_H_1 = None
        self.precomped_H_2 = None
        self.source_amp = None

        self.phase_shift = phase_shift

        # whether to pass zernike/source amp as layers or divide out manually
        self.manual_aberr_corr = manual_aberr_corr

        # make sure parameters from the model training phase don't update
        if self.zernike_coeffs is not None:
            self.zernike_coeffs.requires_grad = False
        if self.source_amplitude is not None:
            for p in self.source_amplitude.parameters():
                p.requires_grad = False

        # change out the propagation operator
        if proptype == 'ASM':
            self.prop = propagation_ASM
        else:
            ValueError(f'Unsupported prop type {proptype}')

        self.linear_conv = linear_conv

        # set a device for initializing the precomputed objects
        try:
            self.dev = next(self.parameters()).device
        except StopIteration:  # no parameters
            self.dev = torch.device('cpu')

    def forward(self, target_amp, mask_layer):
        # compute some initial phase, convert to real+imag representation
        init_phase = self.initial_phase(target_amp) 
        if torch.is_tensor(self.phase_shift):
            self.phase_shift = self.phase_shift.to(init_phase.device)
        init_phase = (init_phase + self.phase_shift) % (2*torch.pi) - np.pi
        real, imag = utils.polar_to_rect(target_amp, init_phase)
        target_complex = torch.complex(real, imag)
        # print(init_phase.min() % (2*torch.pi), init_phase.max())
        # print('Init phase', init_phase)

        # subtract the additional target field
        target_complex_diff = target_complex

        # precompute the propagation kernel only once
        # print('Lets precompute (pre)!!', self.precomped_H_0)
        if self.precomped_H_0 is None:
            # print('Lets precompute (post)!!')
            self.precomped_H_0 = self.prop(target_complex_diff,
                                         self.feature_size,
                                         self.wavelength,
                                         self.distance - self.offset,
                                         return_H=True,
                                         linear_conv=self.linear_conv)
            self.precomped_H_0 = self.precomped_H_0.to(self.dev).detach()
            self.precomped_H_0.requires_grad = False
        
        if self.precomped_H_1 is None:
            # print('Precomputed H0', self.precomped_H_0, self.distance - self.offset)
            self.precomped_H_1 = self.prop(target_complex_diff,
                                         self.feature_size,
                                         self.wavelength,
                                         self.distance,
                                         return_H=True,
                                         linear_conv=self.linear_conv).to(self.dev).detach()
            self.precomped_H_1 = self.precomped_H_1.to(self.dev).detach()
            self.precomped_H_1.requires_grad = False
        
        if self.precomped_H_2 is None:
            # print('Precomputed H1', self.precomped_H_1, self.distance)
            self.precomped_H_2 = self.prop(target_complex_diff,
                                         self.feature_size,
                                         self.wavelength,
                                         self.distance + self.offset,
                                         return_H=True,
                                         linear_conv=self.linear_conv).to(self.dev).detach()
            self.precomped_H_2 = self.precomped_H_2.to(self.dev).detach()
            self.precomped_H_2.requires_grad = False
            # print('Precomputed H2', self.precomped_H_2, self.distance + self.offset)

        # precompute the source amplitude, only once
        if self.source_amp is None and self.source_amplitude is not None:
            self.source_amp = self.source_amplitude(target_amp)
            self.source_amp = self.source_amp.to(self.dev).detach()
            self.source_amp.requires_grad = False

        # print('Target complex diff', target_complex_diff.shape)
        target_complex_p0 = target_complex_diff * (mask_layer == 0).float()
        target_complex_p1 = target_complex_diff * (mask_layer == 1).float()
        target_complex_p2 = target_complex_diff * (mask_layer == 2).float()
        

        # print('Target complex p0', target_complex_p2)
        # implement the basic propagation to the SLM plane
        slm_naive0 = self.prop(target_complex_p0, self.feature_size,
                              self.wavelength, self.distance - self.offset,
                              precomped_H=self.precomped_H_0,
                              linear_conv=self.linear_conv)
        # print('slm_naive p0', slm_naive0)
        slm_naive1 = self.prop(target_complex_p1, self.feature_size,
                              self.wavelength, self.distance,
                              precomped_H=self.precomped_H_1,
                              linear_conv=self.linear_conv)
        # print('slm_naive p1', slm_naive1)
        slm_naive2 = self.prop(target_complex_p2, self.feature_size,
                              self.wavelength, self.distance + self.offset,
                              precomped_H=self.precomped_H_2,
                              linear_conv=self.linear_conv)
        # print('slm_naive p2', slm_naive2)
        
        slm_naive = slm_naive0 + slm_naive1 + slm_naive2
        # print('SLM Naive', slm_naive)
        # switch to amplitude+phase and apply source amplitude adjustment
        amp, ang = utils.rect_to_polar(slm_naive.real, slm_naive.imag)
        # amp, ang = slm_naive.abs(), slm_naive.angle()  # PyTorch 1.7.0 Complex tensor doesn't support
                                                         # the gradient of angle() currently.

        if self.source_amp is not None and self.manual_aberr_corr:
            amp = amp / self.source_amp


        # note the change to usual complex number stacking!
        # We're making this the channel dim via cat instead of stack
        if (self.zernike is None and self.source_amp is None
                or self.manual_aberr_corr):
            if self.latent_codes is not None:
                slm_amp_phase = torch.cat((amp, ang, self.latent_codes.repeat(amp.shape[0], 1, 1, 1)), -3)
            else:
                slm_amp_phase = torch.cat((amp, ang), -3)
        elif self.zernike is None:
            slm_amp_phase = torch.cat((amp, ang, self.source_amp), -3)
        elif self.source_amp is None:
            slm_amp_phase = torch.cat((amp, ang, self.zernike), -3)
        else:
            slm_amp_phase = torch.cat((amp, ang, self.zernike,
                                        self.source_amp), -3)
        return amp, self.final_phase_only(slm_amp_phase), \
            (target_complex_p0, target_complex_p1, target_complex_p2), slm_naive


    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.zernike is not None:
            slf.zernike = slf.zernike.to(*args, **kwargs)
        if slf.precomped_H_0 is not None:
            slf.precomped_H_0 = slf.precomped_H_0.to(*args, **kwargs)
            slf.precomped_H_1 = slf.precomped_H_1.to(*args, **kwargs)
            slf.precomped_H_2 = slf.precomped_H_2.to(*args, **kwargs)
        if slf.source_amp is not None:
            slf.source_amp = slf.source_amp.to(*args, **kwargs)
        if slf.target_field is not None:
            slf.target_field = slf.target_field.to(*args, **kwargs)
        if slf.latent_codes is not None:
            slf.latent_codes = slf.latent_codes.to(*args, **kwargs)

        # try setting dev based on some parameter, default to cpu
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf


class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d):
        super(InitialPhaseUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               PhaseTanh()]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase


class FinalPhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a naive SLM amplitude and phase"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d, num_in=4):
        super(FinalPhaseOnlyUnet, self).__init__()

        net = [Unet(num_in, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               PhaseTanh()]

        self.net = nn.Sequential(*net)

    def forward(self, amp_phase):
        out_phase = self.net(amp_phase)
        return out_phase


class PhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a target amplitude"""
    def __init__(self, num_down=10, num_features_init=16, norm=nn.BatchNorm2d):
        super(PhaseOnlyUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, 1024,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               PhaseTanh()]

        self.net = nn.Sequential(*net)

    def forward(self, target_amp):
        out_phase = self.net(target_amp)
        return (torch.ones(1), out_phase)


class PhaseTanh(nn.Module):
    def __init__(self, scale=torch.pi):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * torch.tanh(x)
