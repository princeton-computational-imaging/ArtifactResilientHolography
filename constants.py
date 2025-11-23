from pado import mm, um, nm


def get_channel(opt):
    return ('red', 'green', 'blue')[opt.channel]

def get_wavelength(opt):
    return (660 * nm, 520 * nm, 450 * nm)[opt.channel]

from types import SimpleNamespace
def get_constants(opt):
    return SimpleNamespace(
        wavelength = get_wavelength(opt),
        chan_str = get_channel(opt),
        prop_dist = opt.prop_dist * mm,
        prop_dist_offset = opt.prop_dist_offset * mm,
        feature_size = (10.8 * um, 10.8 * um),
        image_res = (800, 1280),
        homography_res = (700, 1180),
        data_path = opt.data_path
    )