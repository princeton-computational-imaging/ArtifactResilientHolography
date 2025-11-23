import argparse
import glob
import torch
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser('RGB merger')
parser.add_argument('file_directory', type=str, help='Diretory file with raw images')
args = parser.parse_args()

def stack_files(color, pattern):
    files = sorted(glob.glob(args.file_directory + color + pattern))
    tensors = [torch.load(file, weights_only=False) for file in files]
    tensor_stacked = np.stack(tensors, axis=0)
    return files, tensor_stacked

red_files, red_tensors = stack_files('red', pattern="_ttmx*.pt")
green_files, green_tensors = stack_files('green', pattern="_ttmx*.pt")
blue_files, blue_tensors = stack_files('blue', pattern="_ttmx*.pt")

files = [file.replace("red", "rgb").replace(".pt", ".png") for file in red_files]
stacked_tensors = np.stack((red_tensors, green_tensors, blue_tensors), axis=1)

def save_rgb(rgb_image, filename):
    rgb_image = (rgb_image * 255).clip(0, 255).astype('uint8')
    image = Image.fromarray(rgb_image)
    image.save(filename)

def save_rgba(rgb_image, alpha, filename):
    alpha_image = np.expand_dims(alpha, axis=-1)
    rgba_image = np.concatenate((rgb_image, alpha_image), axis=-1)
    rgba_image = (rgba_image * 255).clip(0, 255).astype('uint8')
    image = Image.fromarray(rgba_image, 'RGBA')
    image.save(filename)

for i, filename in enumerate(files):
    if "wf" in files:
        continue
    print(f'Saving "{filename}"')
    rgb_image = np.moveaxis(stacked_tensors[i,...], 0, -1)
    save_rgb(rgb_image, filename)

red_wfs, red_wfs_tensors  = stack_files('red', pattern="_wf*.pt")
green_wfs, green_wfs_tensors  = stack_files('green', pattern="_wf*.pt")
blue_wfs, blue_wfs_tensors  = stack_files('blue', pattern="_wf*.pt")

if len(red_wfs) == 1:
    r, g, b = red_wfs_tensors[0], green_wfs_tensors[0], blue_wfs_tensors[0]
    for i in range(r.shape[0]):
        abs = np.stack((np.abs(r[i,0,...]), np.abs(g[i,0,...]), np.abs(b[i,0,...])), axis=-1)
        phase = np.stack((np.angle(r[i,0,...]), np.angle(g[i,0,...]), np.angle(b[i,0,...])), axis=-1)
        save_rgb(abs, f"{args.file_directory}/rgb_wf{i}_abs.png")
        save_rgb(phase, f"{args.file_directory}/rgb_wf{i}_phase.png")
else:
    layers = torch.load(args.file_directory +"layer_Whiskey.mat.pt", weights_only=False)

    def plot_layer_i(r, g, b, layers, i):
        abs = np.stack((np.abs(r[i,0,0,...]), np.abs(g[i,0,0,...]), np.abs(b[i,0,0,...])), axis=-1)
        phase = np.stack((np.angle(r[i,0,0,...]), np.angle(g[i,0,0,...]), np.angle(b[i,0,0,...])), axis=-1)
        save_rgba(abs, (layers==i).astype(float), f"{args.file_directory}/rgb_abs{i}.png")
        save_rgba(phase, (layers==i).astype(float), f"{args.file_directory}/rgb_phase{i}.png")

    plot_layer_i(red_wfs_tensors, green_wfs_tensors, blue_wfs_tensors, layers, 0)
    plot_layer_i(red_wfs_tensors, green_wfs_tensors, blue_wfs_tensors, layers, 1)
    plot_layer_i(red_wfs_tensors, green_wfs_tensors, blue_wfs_tensors, layers, 2)

# slm_naive_red = torch.load(args.file_directory +"red_slm_naive.pt", weights_only=False) 
# slm_naive_green = torch.load(args.file_directory +"green_slm_naive.pt", weights_only=False) 
# slm_naive_blue = torch.load(args.file_directory +"blue_slm_naive.pt", weights_only=False)
# slm_naive = np.stack((slm_naive_red, slm_naive_green, slm_naive_blue), axis=-1)[0,0,...]
# save_rgb(np.abs(slm_naive), f"{args.file_directory}/rgb_slm_naive_abs.png")
# save_rgb(np.angle(slm_naive), f"{args.file_directory}/rgb_slm_naive_phase.png")


phase_red = torch.load(args.file_directory +"red_phase.pt", weights_only=False).detach().cpu().numpy()
phase_green = torch.load(args.file_directory +"green_phase.pt", weights_only=False).detach().cpu().numpy()
phase_blue = torch.load(args.file_directory +"blue_phase.pt", weights_only=False).detach().cpu().numpy()
phase_rgb = np.stack((phase_red, phase_green, phase_blue), axis=-1)[0,0,...]
save_rgb(phase_rgb, f"{args.file_directory}/rgb_phase.png")