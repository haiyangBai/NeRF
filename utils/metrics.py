
import torch

def calc_mse(x, y):
    """ Calculates the mean square error between tensors `x` and `y`. """
    return torch.mean((x - y) ** 2)

def calc_psnr(x, y):
    """ Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`. """
    mse = calc_mse(x, y)
    psrn = -10.0 * torch.log10(mse)
    return psrn

def sample_images_at_mc_locs(target_images, sampled_rays_xy):
    """
    Given a set of pixel locations `sampled_rays_xy` this method samples the tensor
    `target_images` at the respective 2D locations.

    This function is used in order to extract the colors from ground truth images
    that correspond to the colors rendered using a Monte Carlo rendering.

    Args:
        target_images: A tensor of shape `(batch_size, ..., 3)`.
        sampled_rays_xy: A tensor of shape `(batch_size, S_1, ..., S_N, 2)`.

    Returns:
        images_sampled: A tensor of shape `(batch_size, S_1, ..., S_N, 3)`
            containing `target_images` sampled at `sampled_rays_xy`.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]

    xy_sample = -sampled_rays_xy.view(ba, -1, 1, 2).clone()

    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2),
        xy_sample,
        align_corners=True,
        mode='bilinear'
    )

    return images_sampled.permute(0, 2, 3, 1).view(ba, *spatial_size, dim)
