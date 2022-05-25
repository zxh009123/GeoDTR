import os
import time
import random
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from .layoutsim import HFlip, Rotate


class USADataset(Dataset):
    def __init__(
        self, 
        data_dir = "../scratch/CVUSA/dataset/", 
        layout_simulation = 'strong', 
        sematic_aug = 'strong',
        mode = 'train', 
        is_polar = True
    ):
        self.data_dir = data_dir

        STREET_IMG_WIDTH = 671
        STREET_IMG_HEIGHT = 122

        self.is_polar = is_polar
        self.mode = mode

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
        self.transforms_street = transforms.Compose(transforms_street)

        self.layout_simulation = layout_simulation

        if mode == "val" or mode == "dev":
            self.file = os.path.join(self.data_dir, "splits", "val-19zl.csv")
        elif mode == "train":
            self.file = os.path.join(self.data_dir, "splits", "train-19zl.csv")
        else:
            raise RuntimeError("no such mode")

        self.data_list = []
        
        csv_file = open(self.file)
        for l in csv_file.readlines():
            data = l.strip().split(",")
            data.pop(2)
            if is_polar:
                data[0] = data[0].replace("bingmap", "polarmap")
                data[0] = data[0].replace("jpg", "png")
            self.data_list.append(data)

        csv_file.close()

        if mode == "dev":
            self.data_list = self.data_list[0:200]


    def __getitem__(self, index):
        satellite_file, ground_file = self.data_list[index]

        satellite = Image.open(os.path.join(self.data_dir, satellite_file))
        ground = Image.open(os.path.join(self.data_dir, ground_file))

        satellite = self.transforms_sat(satellite)
        ground = self.transforms_street(ground)

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

        return {'satellite':satellite, 'ground':ground}


    def __len__(self):
        return len(self.data_list)



if __name__ == "__main__":

    STREET_IMG_WIDTH = 671
    STREET_IMG_HEIGHT = 122
    # SATELLITE_IMG_WIDTH = 256
    # SATELLITE_IMG_HEIGHT = 256
    SATELLITE_IMG_WIDTH = 671
    SATELLITE_IMG_HEIGHT = 122

    transforms_sate = [transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH)),
                    transforms.ToTensor()
                    ]
    transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                    transforms.ToTensor()
                    ]

    dataloader = DataLoader(
        USADataset(
            data_dir = '../scratch/CVUSA/dataset/',
            layout_simulation = 'strong', 
            sematic_aug = 'strong', 
            mode = 'train', 
            is_polar = True),
        batch_size = 4, 
        shuffle = True, 
        num_workers = 8)
    
    total_time = 0
    start = time.time()
    for i,b in enumerate(dataloader):
        end = time.time()
        elapse = end - start
        print("===========================")
        print(b["ground"].shape)
        print(b["satellite"].shape)
        print("===========================")
        time.sleep(2)

        grd = b["ground"][0]
        sat = b["satellite"][0]

        sat = sat * 0.5 + 0.5
        grd = grd * 0.5 + 0.5

        torchvision.utils.save_image(sat, "sat_flip.png")
        torchvision.utils.save_image(grd, "grd_flip.png")

        if i == 2:
            break

    print(total_time / i)
