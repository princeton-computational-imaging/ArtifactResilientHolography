import torch
from pado import nm

def get_lut(wavelength, device):
    # units
    luts = {660*nm:torch.tensor([0.0000, 0.0468, 0.0957, 0.1967, 0.2609, 0.3387, 0.5271, 0.8085, 1.5969,
            1.7280, 1.9710, 2.2918, 2.7944, 3.1100, 3.7153, 4.3704],),
            520*nm: torch.tensor([0.0000, 0.0666, 0.1362, 0.2799, 0.3714, 0.4821, 0.7502, 1.1508, 2.2730,
            2.4597, 2.8055, 3.2621, 3.9775, 4.4266, 5.2882, 6.2207]),
            450*nm: torch.tensor([0.0000,  0.0806,  0.1650,  0.3391,  0.4499,  0.5840,  0.9088,  1.3942,
            2.7536,  2.9797,  3.3987,  3.9519,  4.8185,  5.3626,  6.4063,  7.5360])}
    lut = luts[wavelength].to(device)
    return lut

# t is phase pattern
# d is look up table
def phase_to_bit(t,d):
    # Flatten t to 1D for easier broadcasting
    t_flat = t % (2 * torch.pi)  # shape: (N,)
    d = d % (2 * torch.pi)
    diff = torch.abs(t_flat.unsqueeze(0) - d[:,None,None,None,None,None,None])  # shape: (N, len(d))

    # Get closest indices
    closest_indices = torch.argmin(diff, dim=0)  # shape: (N,)

    # Reshape back to original shape of t
    result = closest_indices.reshape(t.shape)
    return result.float()

def bit_to_phase(t,d):
    return d.flatten()[t.int()] % (2*torch.pi)

def quantize(t,d):
    q = phase_to_bit(t,d)
    return bit_to_phase(q,d)
