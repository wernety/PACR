import torch
'''
训练过程中位置的 XX 的训练
'''
def compute_locations(features,stride):
    h, w = features.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride,
        features.device
    )
    return locations_per_level


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )   # [0,8 ...,192]
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))  # x【0，8，16】*25   y 【0，0，0】
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride + 3*stride  # (size_z-1)/2*size_z 28
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride
    locations = torch.stack((shift_x, shift_y), dim=1) + 32  #alex:48 // 32
    return locations



