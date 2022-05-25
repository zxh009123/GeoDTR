import torch


def HFlip(sat, grd):
    sat = torch.flip(sat, [2])
    grd = torch.flip(grd, [2])

    return sat, grd

def Rotate(sat, grd, orientation, is_polar):
    height, width = grd.shape[1], grd.shape[2]
    if orientation == 'left':
        if is_polar:
            left_sat = sat[:, :, 0:int(width * 0.75)]
            right_sat = sat[:, :, int(width * 0.75):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, -1, [1, 2])
        left_grd = grd[:, :, 0:int(width * 0.75)]
        right_grd = grd[:, :, int(width * 0.75):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'right':
        if is_polar:
            left_sat = sat[:, :, 0:int(width * 0.25)]
            right_sat = sat[:, :, int(width * 0.25):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
        left_grd = grd[:, :, 0:int(width * 0.25)]
        right_grd = grd[:, :, int(width * 0.25):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'back':
        if is_polar:
            left_sat = sat[:, :, 0:int(width * 0.5)]
            right_sat = sat[:, :, int(width * 0.5):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
            sat_rotate = torch.rot90(sat_rotate, 1, [1,2])
        left_grd = grd[:, :, 0:int(width * 0.5)]
        right_grd = grd[:, :, int(width * 0.5):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)
    
    else:
        raise RuntimeError(f"Orientation {orientation} is not implemented")

    return sat_rotate, grd_rotate