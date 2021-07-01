import os
import os.path
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import warnings
warnings.filterwarnings('ignore')
IMG_TRAIN_PATH = 'data/train'

class MaskDataset(Dataset):
    def __init__(self, path, mode):
        self.objects = []
        self.phase = mode
        mean, std = find_mean_var_for_img()
        self.img_size = 224
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5244, 0.4904, 0.4781], std=[0.2655, 0.2623, 0.2576])
        ])
        # if mode == 'train':
        #     transform = transforms.Compose([
        #     transforms.Resize(224) ,
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # else:
        #     transform = transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        for img in path.iterdir():
            if not img.is_file():
                continue
            img_id, bbox, is_mask = img.stem.split('__')
            image = transform(Image.open(img))
            bbox = torch.tensor(json.loads(bbox), dtype=torch.long)
            # if i is negative
            for i in bbox:
                if i < 0:
                    flag_pass_img = True

            # correct w and h if out of boundaries
            img_width, img_height = self.img_size, self.img_size
            if bbox[0] + bbox[2] > img_width:
                bbox[2] = img_width - bbox[0]
            if bbox[1] + bbox[3] > img_height:
                bbox[3] = img_height - bbox[1]

            # if area is 0
            if bbox[2] == 0 or bbox[3] == 0:
                if self.phase == 'train':
                    flag_pass_img = True
                else:
                    bbox[0], bbox[1], bbox[2], bbox[3] = correct_line_to_bbox(bbox[0], bbox[1], bbox[2], bbox[3],
                                                                              img_width, img_height)

            if is_mask == 'True':
                is_mask = 1
            else:
                is_mask = 0
            self.objects.append((image, bbox, is_mask))

    def __getitem__(self, item):
        return self.objects[item]

    def __len__(self):
        return len(self.objects)


def find_mean_var_for_img():
    mean_list = []
    std_list = []
    for img_name in Path(IMG_TRAIN_PATH).iterdir():
        if not img_name.is_file():
            continue
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        img = transform(Image.open(img_name))
        mean_list.append(torch.mean(img.view(3, -1), dim=1).unsqueeze(0))
        std_list.append(torch.std(img.view(3, -1), dim=1).unsqueeze(0))
    mean = torch.mean(torch.cat(mean_list), dim=0)
    std = torch.mean(torch.cat(std_list), dim=0)
    return mean, std


def correct_line_to_bbox(x, y, w, h, img_w, img_h):
    if x>=w:
        if x < img_w:
            w = x + 1
        else:
            x = w - 1
    if y>=h:
        if y<img_h:
            h = y+1
        else:
            y = h - 1
    return [x,y,w,h]