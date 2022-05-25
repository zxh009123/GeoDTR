import os
import random
import numpy as np

from PIL import Image
import scipy.io as sio

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from .layoutsim import HFlip, Rotate


ACT_DATA_MAT_PATH = 'scratch/CVACT/ACT_data.mat'


class ACTDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        layout_simulation = 'strong', 
        sematic_aug = 'strong', 
        is_polar = True, 
        mode = 'train'
    ):
        self.mode = mode
        if mode == 'train':
            folder_name = 'ANU_data_small'
        elif mode == 'val' or mode == 'test':
            folder_name = 'ANU_data_test'
        else:
            raise RuntimeError(f'no such mode: {mode}')
        self.img_root = data_dir

        self.is_polar = is_polar

        STREET_IMG_WIDTH = 671
        STREET_IMG_HEIGHT = 122

        if not is_polar:
            SATELLITE_IMG_WIDTH = 256
            SATELLITE_IMG_HEIGHT = 256
        else:
            SATELLITE_IMG_WIDTH = 671
            SATELLITE_IMG_HEIGHT = 122

        transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH))]
        transforms_sat = [transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH))]

        if sematic_aug == 'strong':
            transforms_sat.append(transforms.ColorJitter(0.3, 0.3, 0.3))
            transforms_street.append(transforms.ColorJitter(0.3, 0.3, 0.3))

            transforms_sat.append(transforms.RandomGrayscale(p=0.2))
            transforms_street.append(transforms.RandomGrayscale(p=0.2))

            transforms_sat.append(transforms.RandomPosterize(p=0.2, bits=4))
            transforms_street.append(transforms.RandomPosterize(p=0.2, bits=4))

            transforms_sat.append(transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)))
            transforms_street.append(transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)))

        elif sematic_aug == 'weak':
            transforms_sat.append(transforms.ColorJitter(0.1, 0.1, 0.1))
            transforms_street.append(transforms.ColorJitter(0.1, 0.1, 0.1))

            transforms_sat.append(transforms.RandomGrayscale(p=0.1))
            transforms_street.append(transforms.RandomGrayscale(p=0.1))

        elif sematic_aug == 'none':
            pass
        else:
            raise RuntimeError(f"sematic augmentation {sematic_aug} is not implemented")

        transforms_sat.append(transforms.ToTensor())
        transforms_sat.append(transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)))

        transforms_street.append(transforms.ToTensor())
        transforms_street.append(transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)))

        self.transforms_sat = transforms.Compose(transforms_sat)
        self.transforms_grd = transforms.Compose(transforms_street)

        self.layout_simulation = layout_simulation

        self.allDataList = ACT_DATA_MAT_PATH

        id_alllist = []
        id_idx_alllist = []

        # load the mat
        anuData = sio.loadmat(self.allDataList)

        idx = 0
        for i in range(0,len(anuData['panoIds'])):
            
            grd_id_align = os.path.join(self.img_root, folder_name, 'streetview_processed', anuData['panoIds'][i] + '_grdView.png')
            if is_polar:
                sat_id_ori = os.path.join(self.img_root, folder_name, 'polarmap', anuData['panoIds'][i] + '_satView_polish.png')
            else:
                sat_id_ori = os.path.join(self.img_root, folder_name, 'satview_polish', anuData['panoIds'][i] + '_satView_polish.jpg')
            id_alllist.append([ grd_id_align, sat_id_ori, anuData['utm'][i][0], anuData['utm'][i][1]])
            id_idx_alllist.append(idx)
            idx += 1

        all_data_size = len(id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', all_data_size)

        if mode == 'val':
            inds = anuData['valSet']['valInd'][0][0] - 1
        elif mode == 'test':
            inds = anuData['valSetAll']['valInd'][0][0] - 1
        elif mode == 'train':
            inds = anuData['trainSet']['trainInd'][0][0] - 1
        Num = len(inds)
        print('Number of samples:', Num)
        self.List = []
        self.IdList = []

        for k in range(Num):
            self.List.append(id_alllist[inds[k][0]])
            self.IdList.append(k)


    def __getitem__(self, idx):
        ground = Image.open(self.List[idx][0])
        ground = self.transforms_grd(ground)

        satellite = Image.open(self.List[idx][1])
        satellite = self.transforms_sat(satellite)

        utm = np.array([self.List[idx][2], self.List[idx][3]])

        #geometric transform
        if self.layout_simulation == "strong":
            hflip = random.randint(0,1)
            if hflip == 1:
                satellite, ground = HFlip(satellite, ground)
            else:
                pass

            orientation = random.choice(["left", "right", "back", "none"])
            if orientation == "none":
                pass
            else:
                satellite, ground = Rotate(satellite, ground, orientation, self.is_polar)

        elif self.layout_simulation == "weak":
            hflip = random.randint(0,1)
            if hflip == 1:
                satellite, ground = HFlip(satellite, ground)
            else:
                pass

        elif self.layout_simulation == "none":
            pass

        else:
            raise RuntimeError(f"layout simulation {self.layout_simulation} is not implemented")

        # return x, y
        return {'satellite':satellite, 'ground':ground, 'utm':utm}


    def __len__(self):
        return len(self.List)



if __name__ == "__main__":
    transforms_sat = [transforms.Resize((122, 671)),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]
    transforms_grd = [transforms.Resize((122, 671)),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]
    dataloader = DataLoader(
        ACTDataset(
            data_dir='../scratch/CVACT/', 
            layout_simulation='none', 
            sematic_aug='none', 
            is_polar=True, 
            mode='train'),
        batch_size=4, 
        shuffle=False, 
        num_workers=8)

    i = 0
    for k in dataloader:
        i += 1
        print("---batch---")
        print("satellite : ", k['satellite'][0,:,1,1])
        print("grd : ", k['ground'][0,:,1,1])
        print("utm : ", k['utm'])
        print("-----------")
        if i > 2:
            break
    