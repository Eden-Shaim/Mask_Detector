"""
Here, we create a custom dataset
"""
import torch
from typing import Any, Tuple, Dict, List
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from torch.utils.data import Dataset


def prep_box_to_rcnn(box):
    xmin, ymin, w, h = box
    xmax, ymax = xmin+w, ymin+h
    return [xmin, ymin, xmax, ymax]

DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
    ])
}


def correct_line_to_bbox(x, y, w, h, img_w, img_h):
    if x >= w:
        if x < img_w:
            w = x + 1
        else:
            x = w - 1
    if y >= h:
        if y < img_h:
            h = y + 1
        else:
            y = h - 1
    return [x, y, w, h]


class MaskDataset(Dataset):
    def __init__(self, image_dir, phase='train'):
        """
        Args:
            image_dir (string): Directory with all the images.
            img_size (int): max image size- weight and height (equal).
            phase (string): train or eval (For transformations).
        """
        self.image_dir = image_dir
        self.phase = phase
        self.transform = DATA_TRANSFORMS[self.phase]
        self.img_size = 224
        self.filenames = sorted(os.listdir(image_dir))
        self.data = self.get_entries()
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id, image, bbox, proper_mask = self.data[idx]

        # create targets
        target = {}
        target['boxes'] = bbox
        target['labels'] = proper_mask

        return image, target

    def get_entries(self) -> List:
        """
        Parse a directory with images.
        :param image_dir: Path to directory with images.
        :return: A list with (image_id, image (as tensor), bbox, proper_mask) for every image in the image_dir.
        """
        example_filenames = os.listdir(self.image_dir)
        data = []
        for filename in self.filenames:
            image_id, bbox, proper_mask = filename.strip(".jpg").split("__")

            # check validity of bbox
            bbox = json.loads(bbox)  # convert string '[x,y,w,h]' to list [x,y,w,h]
            x, y, w, h = bbox
            if x < 0 or y < 0 or w < 0 or h < 0 or w + h == 0:  # skip image if box is 0 or negative w/h
                continue

            # Load image
            image = Image.open(os.path.join(self.image_dir, filename))
            img_width, img_height = image.size
            image = self.transform(image)  # toTensor- convert PIL image to tensor floats in range [0,1]

            # correct w and h if out of boundaries
            if x + w > img_width:
                w = img_width - x
            if y + h > img_height:
                h = img_height - y

            # deal with bboxes where x=w or y=h
            if w == 0 or h == 0:
                if self.phase == 'train':
                    continue
                else:  # eval dataset
                    x, y, w, h = correct_line_to_bbox(x, y, w, h, img_width, img_height)

            # change bbox from [xmin, ymin, w, h] to [xmin, ymin, xmax, ymax]
            bbox = prep_box_to_rcnn([x, y, w, h])

            bbox = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)  # convert shape from [4] to [1,4]

            proper_mask = 2 if proper_mask.lower() == "true" else 1
            proper_mask = torch.tensor([proper_mask], dtype=torch.int64)  # notice it is of shape [1] instead of []

            data.append((image_id, image, bbox, proper_mask))
        return data
