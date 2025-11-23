import torch
import torch.nn.functional as F
import numpy as np


def create_gaussian_window( window_size: int, sigma: float, device):
    # Create a 1D Gaussian kernel
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # Create 2D Gaussian window using outer product
    window = g[:, None] * g[None, :]
    return window.unsqueeze(0).unsqueeze(0)

def rayleigh_distance_loss(field, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    """
    Computes the Rayleigh distance loss between two complex fields.
    
    Args:
        field: Complex tensor of shape (B, 1, H, W)
        window_size: size of Gaussian filter kernel
        sigma: standard deviation of the Gaussian
        C1, C2: constants to stabilize the division when denominator is small

    Returns:
        Scalar tensor: mean Rayleigh distance over the batch
    """
    # img1 is eyebox
    H,W = field.shape[-2:]
    fftfield = torch.fft.fft2(field)
    eyebox = torch.fft.fftshift(fftfield)
    eyebox = eyebox.abs()
    eyebox = eyebox.float()

    # Calculate Target Rayleigh Distribution μ and σ
    # distribution of real/imag components of random phase profile.
    primal_real_imag_mean = 0; 
    primal_real_imag_var = 1/2;

    # real imag components of fourier domain should be gaussian
    fourier_real_imag_mean = 0.0; 
    # fourier_real_imag_var = H * W * primal_real_imag_var; 
    fourier_real_imag_var = primal_real_imag_var * H * W;  

    # mean and variance of intensity of fourier domain (our target rayleigh distribution)
    target_mean = np.sqrt(np.pi * 0.5 * fourier_real_imag_var); 
    target_var = (4.0 - np.pi) * 0.5 * fourier_real_imag_var; 

    # normalize eyebox and target for better performance
    rayleigh_scale = target_mean / np.sqrt(np.pi/2)
    approxmax = rayleigh_scale * np.sqrt(-2*np.log(1/(H*W)))
    target_mean = target_mean / approxmax
    target_var = target_var / approxmax**2
    eyebox = eyebox / eyebox.max()

    
    device = eyebox.device
    # Create Gaussian filter window
    window = create_gaussian_window(window_size, sigma, device=device)
    # Compute local means using Gaussian blur
    mu1 = F.conv2d(eyebox, window, padding=window_size // 2, groups=1)  # local mean of eyebox
    mu2 = target_mean  # local mean of img2

    # Compute squares of means
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(eyebox * eyebox, window, padding=window_size // 2, groups=1) - mu1_sq  # variance of eyebox
    sigma2_sq = target_var  # variance of img2
    sigma12 = 0  # covariance between eyebox and img2

    # Compute Rayleigh distance
    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)  # numerator
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)  # denominator

    # Final Rayleigh distance map
    ssim_map = ssim_n / ssim_d

    # Return average Rayleigh distance
    return -1*torch.log(ssim_map.mean() + 1e-7)


def uniform_distance_loss(slm):
    # squared difference of sorted values as compared to the reference distribution
    wrappedslm = slm % (2*torch.pi)
    reference_range = torch.linspace(0, 2*torch.pi, wrappedslm.numel(), device = slm.device)
    return torch.mean(torch.square(torch.sort(wrappedslm.flatten())[0] - reference_range))

def loss_mix(losses, lambdas = [1.,1.,1.]):
    loss = 0
    for l,a in zip(losses,lambdas):
        loss = loss + l*a
    return loss


def process_dc_block(slm_field, flag_filter):
    if flag_filter:
        current_field = slm_field
        # Compute the 2D FFT
        fft_image = torch.fft.fft2(current_field)
        # Shift the zero-frequency component to the center
        fft_image_shifted = torch.fft.fftshift(fft_image)
        # Block DC term
        n = fft_image_shifted.shape[-1] // 2
        m = fft_image_shifted.shape[-2] // 2

        k = 11
        fft_image_shifted[..., m-k:m+k, n-k:n+k] = 0
        # print(fft_image_shifted.shape)
        # Unshift the zero-frequency component to the center
        fft_image_rec = torch.fft.ifftshift(fft_image_shifted)
        # Recover the phase with ifft
        current_field = torch.fft.ifft2(fft_image_rec)
        # Return new blocked field
        return current_field
    else:
        return slm_field


