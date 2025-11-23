import os
import pado
import torch
from datetime import datetime
import utils.utils as utils
import matplotlib.pyplot as plt
from realtimearh3d import *
import torch.nn.functional as F
import time

import holo_utils as hu
import parser
import quantization
import preprocessing_3d as p3d
from constants import get_constants
import numpy as np


def phase_gen(target_amp, mask_layer, phase_generator, opt, holo_params, lut):
    target_amp = target_amp.to(opt.device)
    mask_layer = mask_layer.to(opt.device)

    prop = pado.propagator.Propagator("ASM")

    # forward model
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=opt.use_fp16):
            torch.cuda.synchronize()
            start_time = time.time()
            slm_amp, slm_phase, wavefronts, slm_naive = phase_generator(target_amp, mask_layer)
            torch.cuda.synchronize()
            end_time = time.time()
    slm_phase = slm_phase.to(torch.float32)


    torch.save(wavefronts[0].detach().cpu().numpy(), f'{opt.output_dir}/{holo_params.chan_str}_wf0.pt')
    torch.save(wavefronts[1].detach().cpu().numpy(), f'{opt.output_dir}/{holo_params.chan_str}_wf1.pt')
    torch.save(wavefronts[2].detach().cpu().numpy(), f'{opt.output_dir}/{holo_params.chan_str}_wf2.pt')
    torch.save(slm_naive.detach().cpu().numpy(), f'{opt.output_dir}/{holo_params.chan_str}_slm_naive.pt')

    if opt.quantize:
        slm_phase = quantization.quantize(slm_phase, lut)

    field = pado.light.Light((opt.batch_size,1,holo_params.image_res[0],holo_params.image_res[1]), holo_params.feature_size[0], holo_params.wavelength, device=opt.device)
    field.set_phase(slm_phase)
    field.set_amplitude_ones()
    final_amp = torch.zeros_like(slm_phase)
    output_amps = []
    prop_dists = [holo_params.prop_dist - holo_params.prop_dist_offset,
                  holo_params.prop_dist,
                  holo_params.prop_dist + holo_params.prop_dist_offset]

    for pid, prop_dist in enumerate(prop_dists):
        output_complex = prop.forward(field,prop_dist).field

        output_complex = hu.process_dc_block(output_complex, opt.set_dc_block)

        output_lin_intensity = torch.sum(output_complex.abs()**2 * opt.scale_output, dim=1, keepdim=True)

        output_amp = torch.pow(output_lin_intensity, 0.5)
        output_amps.append(output_amp)

        final_amp += output_amp * (mask_layer == pid)

    # crop outputs to the region we care about
    amplitude_scaling = target_amp.mean(dim=(2,3), keepdim=True) / output_amp.mean(dim=(2,3), keepdim=True)
    final_amp = final_amp * amplitude_scaling
    final_amp = utils.crop_image(final_amp, holo_params.homography_res, stacked_complex=False)
    output_amps = [oamp * amplitude_scaling for oamp in output_amps]
    output_ints = [torch.square(oamp) for oamp in output_amps]

    return slm_phase, output_amps, output_ints, output_complex, final_amp, end_time - start_time


def save_images_phases_eyeboxes(output_phase, output_amp, output_complex, final_amp,holo_params):
    # Transpose list of lists
    output_amp = list(zip(*output_amp))

    print('Saving images')
    for l in range(len(output_amp)):
        output_amp_l = torch.cat(output_amp[l], dim = 1)
        for i in range(output_amp_l.shape[0]):
            for j in range(output_amp_l.shape[1]):
                temp = output_amp_l[i, j, ...]
                temp = temp / temp.max()
                temp = temp.cpu().detach().numpy()
                plt.imsave(f'{holo_params.chan_str}_nottmx_{i}_{j}_layer{l}.png', temp)
            output_amp_tmx = output_amp_l[i,...].mean(dim=0).detach().cpu().numpy()
            final_amp_tmx = final_amp[i,...].mean(dim=0).detach().cpu().numpy()
            torch.save(output_amp_tmx, f'{holo_params.chan_str}_ttmx{i}_layer{l}.pt')
            plt.imsave(f'{holo_params.chan_str}_ttmx{i}_layer{l}.png', output_amp_tmx / output_amp_tmx.max())
            plt.imsave(f'{holo_params.chan_str}_ttmx{i}_stack.png', final_amp_tmx / final_amp_tmx.max())

    print('Saving phases, eyeboxes')
    for i in range(output_amp_l.shape[0]):
        for j in range(output_amp_l.shape[1]):
            temp_phase = output_phase[i,j,...] / (2*torch.pi) + 0.5
            
            fftfield = torch.fft.fft2(output_complex[i,j,...])
            eyebox = torch.fft.fftshift(fftfield).abs()
            eyebox /= eyebox.max()
            # temp_complex = .abs() / output_complex[i,j,...].abs().max()
            plt.imsave(f"{holo_params.chan_str}_phase_{i}_{j}.png", temp_phase.detach().cpu().numpy(), cmap='hsv')
            plt.imsave(f"{holo_params.chan_str}_eyebox_{i}_{j}.png", eyebox.detach().cpu().numpy(), cmap='gist_gray')

    torch.save(output_phase, F"{holo_params.chan_str}_phase.pt")

    print('Everything saved!')


def main(opt):
    generators = opt.generator_path.split(',')
    holo_params = get_constants(opt)
    print(f'***********************************')
    print(f'Running test for {holo_params.chan_str} channel')
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    # tensorboard setup and file naming
    time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')

    lut = quantization.get_lut(holo_params.wavelength, opt.device)
    target_amp, mask_layer = p3d.load_3d_data(opt, holo_params)


    print('Generating phase patterns!')
    output_phase = []
    output_amp = []
    output_complex = []
    exec_times = []
    final_amps = []
    for generator in generators:
        phase_generator = RealTimeARH3D(
                distance=holo_params.prop_dist,
                offset=holo_params.prop_dist_offset,
                wavelength=holo_params.wavelength,
                feature_size=holo_params.feature_size,
                initial_phase=InitialPhaseUnet(4, 16),
                final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2),
                proptype=opt.proptype).to(opt.device)
        phase_generator.load_state_dict(torch.load(generator, map_location=opt.device))
        phase_generator.eval()

        phase, temp_amps, temp_ints, complex, final_amp, elapsed_time = \
            phase_gen(target_amp, mask_layer, phase_generator, opt, holo_params, lut)
        output_phase.append(phase)
        output_amp.append(temp_amps)
        output_complex.append(complex)
        exec_times.append(elapsed_time)
        final_amps.append(final_amp)

    output_phase = torch.cat(output_phase, dim = 1)
    output_complex = torch.cat(output_complex, dim = 1)
    final_amps = torch.cat(final_amps, dim = 1)
    exec_times_np = np.array(exec_times)
    exec_time_seconds = np.median(exec_times_np)
    fps = 1 / exec_time_seconds
    print(f'Median exec time, tm=1 (running tm aside), bs={opt.batch_size}: '
        f'{exec_time_seconds:.2f} seconds, {fps:.2f} fps')

    # Saving everything!
    os.chdir(opt.output_dir)
    save_images_phases_eyeboxes(output_phase, output_amp, output_complex, final_amps, holo_params)


if __name__ == '__main__':    # Command line argument processing
    p = parser.create_parser()
    parser.add_inference_args(p)
    opt = p.parse_args()
    main(opt)