
from scipy.io import loadmat
from skimage.transform import resize
from utils.augmented_image_loader import ImageLoader
import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def preprocess_splitlohmann(path, opt, holo_params):
    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    diopter_map = resize(data['diopterMap'], holo_params.image_res)
    image = resize(data['textureMap'], holo_params.image_res)[...,opt.channel]
    t1 = np.quantile(diopter_map, 0.33)
    t2 = np.quantile(diopter_map, 0.66)
    layer = np.ones_like(diopter_map, dtype=np.int32)
    layer[diopter_map<t1] = 0
    layer[diopter_map>t2] = 2
    bn = os.path.basename(path)
    plt.imsave(opt.output_dir + 'diopter_map_' + bn + '.png', diopter_map)
    plt.imsave(opt.output_dir + 'texture_map_' + bn + '.png', image)
    plt.imsave(opt.output_dir + 'layer_' + bn + '.png', layer)
    torch.save(layer, opt.output_dir + 'layer_' + bn + '.pt')
    return torch.tensor(image).to(torch.float32), torch.tensor(layer)


def load_3d_data(opt, holo_params):
    data_path = opt.data_path
    # Data loading
    if data_path == 'data/pics':
        print('Loading standard dataset')
        image_loader = ImageLoader(data_path,
                                channel=opt.channel,
                                batch_size=opt.batch_size,
                                image_res=holo_params.image_res,
                                homography_res=holo_params.homography_res,
                                shuffle=False,
                                vertical_flips=False,
                                horizontal_flips=False)


        for _, target in enumerate(image_loader):
            sample = target
            break
        target_amp, target_res, target_filename = sample
        mask_layer = torch.ones_like(target_amp, dtype=torch.int32)
        B, _, H, W = target_amp.shape
        mask_layer[:,:,:,:W//3] = 0
        mask_layer[:,:,:,2*W//3:] = 2
        print(target_amp.shape, mask_layer.shape, type(target_amp), type(mask_layer))
    elif data_path == 'data/scenes_splitlohmann':
        motorcycle, motorcycle_layer = preprocess_splitlohmann(data_path + '/mat/Motorcycle.mat', opt, holo_params)
        whiskey, whiskey_layer = preprocess_splitlohmann(data_path + '/mat/Whiskey.mat', opt, holo_params)
        if opt.batch_size == 1:
            target_amp = whiskey.unsqueeze(0).unsqueeze(0)
            mask_layer = whiskey_layer.unsqueeze(0).unsqueeze(0)
        elif opt.batch_size == 2:
            target_amp = torch.stack((motorcycle, whiskey), dim=0).unsqueeze(1)
            mask_layer = torch.stack((motorcycle_layer, whiskey_layer), dim=0).unsqueeze(1)
        else:
            raise AssertionError(f"Batch size {opt.batch_size} is not supported with Split Lohmann's dataset. Only 1 or 3")
        # print(target_amp.shape, mask_layer.shape, target_amp.dtype, mask_layer.dtype)
    else:
        raise FileNotFoundError(f'Directory not found: {data_path}')
    
    return target_amp, mask_layer
