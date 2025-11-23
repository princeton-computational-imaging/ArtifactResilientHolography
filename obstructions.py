import pado
from pado.math import mm, nm, um
import matplotlib.pyplot as plt
import torch
import numpy as np


def apply_postpupil(
    field,
    z: float = -10 * mm,
    wvl: float = 660 * nm,
    pitch: float = 10.8 * um,
    obs_plane_amp: str = None,
    obs_plane_phase: str = None,
    device: str = 'cuda'
):
    """Apply post-pupil obstruction in the optical pipeline."""
    # Define obstruction plane
    R, C = field.get_field().shape[-2:]
    obstruction_element = pado.optical_element.Aperture(
        (1, 1, R, C), pitch, 0, 'circle', wvl, device=device
    )
    amp_img = plt.imread(obs_plane_amp)
    phase_img = plt.imread(obs_plane_phase)
    amp_tensor = torch.tensor(amp_img, device=device)
    phase_tensor = torch.tensor(phase_img, device=device)

    obstruction_element.set_amplitude_change(
        resample_tensor(amp_tensor, R, C)
    )
    obstruction_element.set_phase_change(
        resample_tensor(phase_tensor, R, C) * 2 * np.pi
    )

    # Propagate, apply element, propagate back
    prop = pado.propagator.Propagator('ASM')
    field = prop.forward(field, z)
    field = obstruction_element.forward(field)
    field = prop.forward(field, -z)
    return field


def apply_prepupil(
    field,
    scalar: float = 0.8,
    z: float = 5 * mm,
    wvl: float = 660 * nm,
    pitch: float = 10.8 * um,
    focal_length: float = 65 * mm,
    obs_plane_amp: str = None,
    obs_plane_phase: str = None,
    device: str = 'cuda'
):
    """Pre-pupil FFT/pad/prop/obstruct/invprop/crop/IFFT optical block."""
    # FFT to frequency domain
    eye_field = pado.math.fft(field.get_field())
    eye_spacing = eyebox_spacing(field, focal_length)

    # Pad to match SLM
    pad_eyefield = pad_eyebox_slm_pado(eye_field, eye_spacing, ratio=scalar)
    R, C = pad_eyefield.shape[-2:]

    # Propagate
    pad_eyefield = holotorch_asm(pad_eyefield, eye_spacing, field.wvl, z)

    # Prepare obstruction
    obs_amp = torch.tensor(plt.imread(obs_plane_amp), device=device)
    obs_phase = torch.tensor(plt.imread(obs_plane_phase), device=device) * 2 * np.pi

    obs_amp = resample_tensor(obs_amp, eye_field.shape[-2], eye_field.shape[-1])
    obs_amp = pad_eyebox_slm_pado(obs_amp, eye_spacing, ratio=scalar)

    obs_phase = resample_tensor(obs_phase, eye_field.shape[-2], eye_field.shape[-1])
    obs_phase = pad_eyebox_slm_pado(obs_phase, eye_spacing, ratio=scalar)

    obstruction_element = obs_amp * torch.exp(1j * obs_phase)

    # Apply obstruction
    pad_eyefield = pad_eyefield * obstruction_element

    # Inverse propagate
    pad_eyefield = holotorch_asm(pad_eyefield, eye_spacing, field.wvl, -z)

    # Crop back to original
    eye_field = center_crop(pad_eyefield, field.shape()[-2], field.shape()[-1])

    # Inverse FFT
    out_field = field.clone()
    out_field.set_field(pado.math.ifft(eye_field))

    return out_field


def create_asmkernel(
    field,
    spacing,
    wvl,
    z
):
    """
    Compute the Angular Spectrum Method (ASM) kernel for free-space propagation.
    Based on:
        https://github.com/facebookresearch/holotorch/blob/45fd4c5b6b6a02fa657fadb2a2efc0110788ca59/holotorch/Optical_Propagators/ASM_Prop.py
    Args:
        field: An object with .wvl, .device, and .shape() attributes
        spacing: Sequence (dx[, dy]) (physical units)
        wvl: Wavelength(s) [scalar or sequence]
        z: Propagation distance
    Returns:
        ASM_Kernel: torch.Tensor
    """
    dx = spacing[0]
    dy = spacing[1] if len(spacing) > 1 else dx

    # Shape handling
    H, W = field.shape[-2], field.shape[-1]

    # Wavelength and spacing handling
    wavelengths = torch.tensor(wvl, device=field.device).reshape(-1)
    wavelengths_TC = wavelengths[:, None]  # (T, 1)

    dx_TC = dx[None, None]  # shape: (1,1)
    dy_TC = dy[None, None]  # shape: (1,1)

    # Spatial frequency grid
    with torch.no_grad():
        kx = torch.linspace(-0.5, 0.5, H, device=field.device)
        ky = torch.linspace(-0.5, 0.5, W, device=field.device)
        Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")

    # Normalize frequencies
    Kx = 2 * np.pi * Kx / dx_TC
    Ky = 2 * np.pi * Ky / dy_TC
    K2 = Kx ** 2 + Ky ** 2

    # ASM kernel
    K_lambda = 2 * np.pi / wavelengths_TC  # (T,1)
    K_lambda_2 = K_lambda ** 2

    ang = z * torch.sqrt(K_lambda_2 - K2)
    if ang.is_complex():
        ang = ang.real

    # Bandlimit the kernel (see Matsushima et al., Opt. Express 17, 19662-19673 (2009))
    length_x = H * dx_TC
    length_y = W * dy_TC
    f_y_max = 2 * np.pi / torch.sqrt((2 * z * (1 / length_x)) ** 2 + 1) / wavelengths_TC
    f_x_max = 2 * np.pi / torch.sqrt((2 * z * (1 / length_y)) ** 2 + 1) / wavelengths_TC

    H_filter = torch.zeros_like(ang)
    valid_mask = (torch.abs(Kx) < f_x_max) & (torch.abs(Ky) < f_y_max)
    H_filter[valid_mask] = 1

    ASM_Kernel = H_filter * torch.exp(1j * H_filter * ang)
    return ASM_Kernel


def holotorch_asm_pado(
    field,
    spacing,
    z
):
    """
    Angular Spectrum Method (ASM) propagation for asymmetric pixel spacings.
    Args:
        field: Optical field obj.
        spacing: (dx, dy)
        z: Propagation distance
    Returns:
        Propagated field obj (same type)
    """
    spacing_tensor = torch.tensor(spacing, device=field.device)
    kernel = create_asmkernel(field.get_field(), spacing_tensor, field.wvl, z).squeeze()
    field_tensor = field.get_field()
    fft_field = pado.math.fft(field_tensor)
    propagated_fft = fft_field * kernel
    propagated_field = pado.math.ifft(propagated_fft)
    result = field.clone()
    result.set_field(propagated_field)
    return result


def holotorch_asm(
    field,
    spacing,
    wvl,
    z
):
    """
    Angular Spectrum Method (ASM) propagation for asymmetric pixel spacings.
    Args:
        field: Tensor
        spacing: (dx, dy)
        wvl: Wavelength(s)
        z: Propagation distance
    Returns:
        propagated tensor
    """
    spacing_tensor = torch.tensor(spacing, device=field.device)
    kernel = create_asmkernel(field, spacing_tensor, wvl, z).squeeze()
    fft_field = pado.math.fft(field)
    propagated_fft = fft_field * kernel
    propagated_field = pado.math.ifft(propagated_fft)
    return propagated_field


def eyebox_spacing(
    field,
    focal_length
):
    """
    Compute output pixel spacing at eyebox for given focal length.
    Args:
        field: Has 'wvl', 'pitch', and 'shape' attributes
        focal_length: Focal length value (float)
    Returns:
        torch.Tensor: Output pixel spacings (dx, dy) at eyebox.
    """
    wvl = field.wvl
    pitch = field.pitch
    device = field.device
    shape = field.shape()[-2:]
    center_wavelength = torch.tensor(wvl, device=device)
    pitch_tensor = torch.tensor([pitch, pitch], device=device)
    shape_tensor = torch.tensor(shape, device=device)
    dx_output = (center_wavelength * abs(focal_length)) / (pitch_tensor * shape_tensor)
    return dx_output


def pad_eyebox_slm_pado(
    eye,
    eye_spacing,
    ratio: float = 1.0
):
    """
    Pad an 'eye' field/tensor to match the SLM resolution.
    Args:
        eye: Field/tensor to pad.
        eye_spacing: Pixel spacing(s) (tuple/list/array)
        ratio: Size scaling factor.
    Returns:
        The padded field/tensor matching SLM resolution.
    """
    # TI PLM panel size: 800 x 1280 pixels, 10.8 um pitch
    slm_pixel_count = torch.tensor([800, 1280])
    pixel_pitch_um = 10.8
    slm_size = slm_pixel_count * pixel_pitch_um * um * ratio

    new_spacing = torch.tensor(eye_spacing)
    target_height = int(slm_size[0].item() // new_spacing[0].item())
    target_width  = int(slm_size[1].item() // new_spacing[1].item())

    padded_eye = pad_to_resolution(eye, target_height, target_width)
    return padded_eye


def pad_to_resolution(
    field,
    target_height: int,
    target_width: int
):
    """
    Pad field tensor to (target_height, target_width), centering content.
    Args:
        field: Tensor (any shape, last two dims = H, W)
        target_height: int
        target_width: int
    Returns:
        Tensor padded to target height/width (centered).
    """
    tensor = field
    current_height, current_width = tensor.shape[-2:]
    pad_height = max(0, target_height - current_height)
    pad_width = max(0, target_width - current_width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_tensor = torch.nn.functional.pad(
        tensor,
        (pad_left, pad_right, pad_top, pad_bottom)
    )
    return padded_tensor


def center_crop(
    tensor,
    H: int,
    W: int
):
    """
    Crop the center (H x W) region from input tensor.
    Args:
        tensor: (..., height, width) shape
        H: int, output height
        W: int, output width
    Returns:
        Cropped tensor (..., H, W)
    """
    shape = tensor.shape
    height = shape[-2]
    width  = shape[-1]

    top = (height - H) // 2
    left = (width - W) // 2
    bottom = top + H
    right = left + W
    return tensor[..., top:bottom, left:right]


def resample_tensor(
    t: torch.Tensor,
    H: int,
    W: int,
    mode: str = 'bilinear',
    align_corners: bool = True
):
    """
    Resample tensor t to arbitrary size (H, W).
    Args:
        t: Input tensor (..., h, w)
        H: Target height
        W: Target width
        mode: Interpolation ('bilinear' default)
        align_corners: For interpolation modes (see PyTorch docs)
    Returns:
        Resampled tensor (..., H, W)
    """
    import torch.nn.functional as F

    t = t.float()
    orig_shape = t.shape

    if len(orig_shape) < 2:
        raise ValueError("Input tensor must have at least 2 spatial dims (h, w)")

    leading = orig_shape[:-2]
    h, w = orig_shape[-2:]
    num_leading = int(np.prod(leading)) if leading else 1

    t_flat = t.reshape(num_leading, h, w)
    t_flat = t_flat.unsqueeze(1)  # shape: [batch, channel=1, h, w]
    resampled = F.interpolate(
        t_flat,
        size=(H, W),
        mode=mode,
        align_corners=align_corners if 'linear' in mode else None
    )
    resampled = resampled.squeeze(1)
    resampled = resampled.reshape(*leading, H, W)
    return resampled

