import os
import pado
import torch
from datetime import datetime
import utils.utils as utils
import matplotlib.pyplot as plt
from holonet import *
from utils.augmented_image_loader import ImageLoader
import time


import holo_utils as hu
import parser
import quantization
from constants import get_constants

from obstructions import *


def phase_gen(sample, phase_generator, opt, holo_params, lut):
    target_amp, _, _ = sample
    target_amp = target_amp.to(opt.device)

    prop = pado.propagator.Propagator("ASM")

    # forward model
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=opt.use_fp16):
            slm_amp, slm_phase, target_complex = phase_generator(target_amp)
    torch.cuda.synchronize()
    end_time = time.time()
    slm_phase = slm_phase.to(torch.float32)
   
    torch.save(target_complex.detach().cpu().numpy(), f'{opt.output_dir}/{holo_params.chan_str}_wf.pt')

    if opt.quantize:
        slm_phase = quantization.quantize(slm_phase, lut)

    field = pado.light.Light((opt.batch_size,1,holo_params.image_res[0],holo_params.image_res[1]), holo_params.feature_size[0], holo_params.wavelength, device=opt.device)
    field.set_phase(slm_phase)
    field.set_amplitude_ones()

    output_complex = prop.forward(field, holo_params.prop_dist)

    if opt.set_prepupil_obstructions:
        output_complex = apply_prepupil(output_complex, wvl = holo_params.wavelength, pitch = holo_params.feature_size[0], scalar = 0.8, z = 10*mm, obs_plane_amp = "data/obstructions/victor_amp.png", obs_plane_phase = "data/obstructions/victor_phase.png")
    if opt.set_postpupil_obstructions:
        output_complex = apply_postpupil(output_complex, wvl = holo_params.wavelength, pitch = holo_params.feature_size[0], obs_plane_amp = "data/obstructions/oscar_amp.png", obs_plane_phase = "data/obstructions/oscar_phase.png")
    output_complex = output_complex.field
    output_complex = hu.process_dc_block(output_complex, opt.set_dc_block)
    output_lin_intensity = torch.sum(output_complex.abs()**2 * opt.scale_output, dim=1, keepdim=True)
    output_amp = torch.pow(output_lin_intensity, 0.5)

    # crop outputs to the region we care about
    output_amp = output_amp * target_amp.mean(dim=(2,3), keepdim=True) / output_amp.mean(dim=(2,3), keepdim=True)
    output_amp = utils.crop_image(output_amp, holo_params.homography_res, stacked_complex=False)

    return slm_phase, output_amp, output_complex, end_time - start_time


def save_amplitude_phases(output_amp, output_phase, output_complex, sample, holo_params):
    target_amp, _, _ = sample
    target_amp = utils.crop_image(target_amp, holo_params.homography_res, stacked_complex=False)
    for i in range(output_amp.shape[0]):
        plt.imsave(f'{holo_params.chan_str}_target_{i}.png', target_amp[i,0])
    
    for i in range(output_amp.shape[0]):
        for j in range(output_amp.shape[1]):
            temp = output_amp[i, j, ...]
            temp = temp / temp.max()
            temp = temp.cpu().detach().numpy()
            temp_phase = output_phase[i,j,...] / (2*torch.pi) + 0.5
            
            oc = output_complex[i,j,...]
            fftfield = torch.fft.fft2(oc)
            eyebox = torch.fft.fftshift(fftfield).abs()
            eyebox /= eyebox.max()
            # temp_complex = .abs() / output_complex[i,j,...].abs().max()
            plt.imsave(f'{holo_params.chan_str}_nottmx_{i}_{j}.png', temp)
            plt.imsave(f"{holo_params.chan_str}_phase_{i}_{j}.png", temp_phase.detach().cpu().numpy(), cmap='hsv')
            torch.save(temp_phase.detach().cpu().numpy(), f"{holo_params.chan_str}_phase_ttmx_{i}_{j}.png")
            plt.imsave(f"{holo_params.chan_str}_phasetarget_{i}_{j}.png", oc.angle().detach().cpu().numpy(), cmap='hsv')
            plt.imsave(f"{holo_params.chan_str}_eyebox_{i}_{j}.png", eyebox.detach().cpu().numpy(), cmap='gist_gray')
        output_amp_tmx = output_amp[i,:,:,:].mean(dim=0).detach().cpu().numpy()
        torch.save(output_amp_tmx, f'{holo_params.chan_str}_ttmx{i}.pt')
        plt.imsave(f'{holo_params.chan_str}_ttmx{i}.png', output_amp_tmx / output_amp_tmx.max())

    torch.save(output_phase, F"{holo_params.chan_str}_phase.pt")
    print('Everything saved!')


def main(opt):
    print('Generator_path', opt.generator_path)
    generators = opt.generator_path.split(',')
    holo_params = get_constants(opt)
    print(f'***********************************')
    print(f'Running test for {holo_params.chan_str} channel')
    # tensorboard setup and file naming
    time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    ###############
    # Load models #
    ###############

    # loads images from disk, set to augment with flipping
    image_loader = ImageLoader(holo_params.data_path,
                            channel=opt.channel,
                            batch_size=opt.batch_size,
                            image_res=holo_params.image_res,
                            homography_res=holo_params.homography_res,
                            shuffle=False,
                            vertical_flips=False,
                            horizontal_flips=False)

    # Get first 
    sample = image_loader.load_batch([(0,None), (1,None), (2,None), (3,None), (4,None)][:opt.batch_size])
    lut = quantization.get_lut(holo_params.wavelength, opt.device)

    print('Generating phase patterns!')
    output_phase = []
    output_amp = []
    output_complex = []
    losses = []
    exec_times = []
    for generator in generators:
        phase_generator = HoloNet(
                distance=holo_params.prop_dist,
                wavelength=holo_params.wavelength,
                zernike_coeffs=None,
                feature_size=holo_params.feature_size,
                source_amplitude=None,
                initial_phase=InitialPhaseUnet(4, 16),
                final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2),
                manual_aberr_corr=opt.manual_aberr_corr,
                target_field=None,
                latent_codes=None,
                inference=True,
                proptype=opt.proptype).to(opt.device)
        phase_generator.load_state_dict(torch.load(generator, map_location=opt.device))
        phase_generator.eval()

        phase, temp_amp, complex, etime = phase_gen(sample, phase_generator, opt, holo_params, lut)
        output_phase.append(phase)
        output_amp.append(temp_amp)
        output_complex.append(complex)
        exec_times.append(etime)
        losses.append(hu.rayleigh_distance_loss(complex))
        print("Rayleigh distance", torch.tensor(losses).mean())

    output_phase = torch.cat(output_phase, dim = 1)
    output_amp = torch.cat(output_amp, dim = 1)
    output_complex = torch.cat(output_complex, dim = 1)
    exec_times_np = np.array(exec_times)
    exec_time_seconds = np.median(exec_times_np)
    fps = 1 / exec_time_seconds
    print(f'Tensor output shape {output_phase.shape}')
    print(f'Median exec time, tm=1 (running tm aside), bs={opt.batch_size}: '
        f'{exec_time_seconds:.2f} seconds, {fps:.2f} fps')
    
    os.chdir(opt.output_dir)
    save_amplitude_phases(output_amp, output_phase, output_complex, sample, holo_params)


if __name__ == '__main__':    # Command line argument processing
    p = parser.create_parser()
    parser.add_inference_args(p)
    opt = p.parse_args()
    main(opt)
