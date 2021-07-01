import argparse
import torch.utils.data
from dataset import MaskDataset
from fasterrcnn_train import collate_fn
import os
import gdown
import warnings
import pandas as pd
import numpy as np
import torchvision
warnings.filterwarnings("ignore")


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

#####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Downloading trained model --->')
module_path = os.path.dirname(os.path.realpath(__file__))
gdrive_file_id = '1Dvake3deIwcE6L7W2PUJB_8Jdvo-PvNM'

url = f'https://drive.google.com/uc?id={gdrive_file_id}'
trained_model = os.path.join(module_path, 'faster_rcnn.pth.tar')
gdown.download(url, trained_model, quiet=False)
print('Downloading trained model <---')


print('Loading model --->')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=3,
                                                                 box_detections_per_img=1, min_size=224, max_size=224)
model.load_state_dict(torch.load(trained_model))
model.to(device)
# model = torch.load(trained_model)
print('Loading model <---')

print('Prepare data --->')
dataset = MaskDataset(image_dir=args.input_folder, phase='test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
print('Prepare data <---')

# Evaluate model on given data
print(f"Evaluating data --->")
model.eval()
bbox_predictions = []
mask_predictions = []
with torch.no_grad():
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predictions = model(images, targets)  # preds: [{boxes, labels, filenames, scores}]
        for prediction in predictions:
            bbox = prediction['boxes']
            if bbox.numel():
                bbox = bbox[0].to('cpu')
                pred_label = prediction['labels'].item()
                pred_label = pred_label - 1
                print(pred_label, bbox)
                pred_label = True if pred_label else False
            else:
                bbox = list(np.random.randint(0, high=224, size=(4)))
                pred_label = np.random.randint(2, size=1).astype(np.bool)[0]
            bbox_predictions.append(bbox)
            mask_predictions.append(pred_label)
        del images, targets
        torch.cuda.empty_cache()

print(f"Evaluating data <---")

prediction_df = pd.DataFrame(columns=['filename', 'h', 'w', 'y', 'x', 'proper_mask'])
for i in range(len(bbox_predictions)):
    xmin, ymin, xmax, ymax = bbox_predictions[i]
    w = (xmax - xmin).item()
    h = (ymax - ymin).item()
    print([files[i], h, w, ymin.item(), xmin.item(), mask_predictions[i]])
    prediction_df.loc[i] = [files[i], h, w, ymin.item(), xmin.item(), mask_predictions[i]]

print(f"Saving predictions --->")
prediction_df.to_csv("prediction.csv", index=False, header=True)
print(f"Saving predictions <---")

