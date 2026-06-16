import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def weighted_charbonnier_loss(network_output, gt, weight=None, eps=1e-3):
    diff = network_output - gt
    loss_map = torch.sqrt(diff * diff + eps * eps)
    if weight is None:
        return loss_map.mean()
    w = weight.to(loss_map.dtype)
    # If `w` is a scalar (e.g. use_attention=False -> weight=1.0),
    # weighted mean should degrade to plain mean instead of summing all pixels.
    if w.numel() == 1:
        return loss_map.mean()
    return (w * loss_map).sum() / (w.sum() + 1e-6)


def _to_nchw(img):
    if img.dim() == 4:
        return img
    if img.dim() == 3:
        return img.unsqueeze(0)
    raise ValueError(f"Unsupported tensor shape for image: {tuple(img.shape)}")


def _normalize_per_image(x, eps=1e-6):
    x_nchw = _to_nchw(x)
    x_min = x_nchw.amin(dim=(2, 3), keepdim=True)
    x_max = x_nchw.amax(dim=(2, 3), keepdim=True)
    x_norm = (x_nchw - x_min) / (x_max - x_min + eps)
    if x.dim() == 3:
        return x_norm.squeeze(0)
    return x_norm


def gaussian_blur(img, kernel_size=7, sigma=1.5):
    x = _to_nchw(img)
    if kernel_size % 2 == 0:
        kernel_size += 1
    device = x.device
    dtype = x.dtype
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / (kernel_1d.sum() + 1e-12)
    c = x.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size).repeat(c, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(x, kernel_x, padding=(0, radius), groups=c)
    x = F.conv2d(x, kernel_y, padding=(radius, 0), groups=c)
    if img.dim() == 3:
        return x.squeeze(0)
    return x


def structural_attention_map(gt, hf_lambda=0.5, blur_kernel=7, blur_sigma=1.5):
    x = _to_nchw(gt)
    device = x.device
    dtype = x.dtype

    c = x.shape[1]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)

    gx = F.conv2d(x, sobel_x, padding=1, groups=c)
    gy = F.conv2d(x, sobel_y, padding=1, groups=c)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-6)

    blur = gaussian_blur(x, kernel_size=blur_kernel, sigma=blur_sigma)
    high = torch.abs(x - blur)

    attn = edge + hf_lambda * high
    attn = _normalize_per_image(attn).clamp(0.0, 1.0)

    if gt.dim() == 3:
        return attn.squeeze(0)
    return attn

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window



def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)



def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

