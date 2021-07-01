import torch
import dataset
import torchvision
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt


def calc_iou(bbox_a, bbox_b):
    """
    Calculate intersection over union (IoU) between two bounding boxes with a (x, y, w, h) format.
    :param bbox_a: Bounding box A. 4-tuple/list.
    :param bbox_b: Bounding box B. 4-tuple/list.
    :return: Intersection over union (IoU) between bbox_a and bbox_b, between 0 and 1.
    """
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection  # Union = Total Area - Intersection
    return intersection / union


def calc_metrics(detections, targets):
    iou = 0
    acc = 0
    for detection, target in zip(detections, targets):
        pred_bbox = detection['boxes']
        pred_label = detection['labels']
        if pred_bbox.numel():
            pred_bbox = pred_bbox[0]
            pred_bbox[2] = pred_bbox[2] - pred_bbox[0]
            pred_bbox[3] = pred_bbox[3] - pred_bbox[1]
            true_bbox = target['boxes'][0].tolist()
            true_bbox[2] = true_bbox[2] - true_bbox[0]
            true_bbox[3] = true_bbox[3] - true_bbox[1]
            iou += calc_iou(pred_bbox, true_bbox)

            if pred_label == target['labels']:
                acc += 1

    if isinstance(iou, torch.Tensor):
        iou = iou.item()
    return iou, acc


def train(model, optimizer, data_loader, test_dataloader, data_types,datasets_sizes, device, epoch, lr_scheduler=None):
    loss_train = []
    scores = {"bbox_iou_train" : [],
              "mask_accuracy_train" : [],
              "bbox_iou_test": [],
              "mask_accuracy_test": [],
              }
    for epoch in range(epoch):
        print(epoch)
        epoch_start = time.time()
        epoch_loss = 0
        for images, targets in data_loader:
            model.train()

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            optimizer.zero_grad()
            losses.backward()
            # Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            epoch_loss += loss_value / len(data_loader)

        if lr_scheduler is not None:
            lr_scheduler.step()
        del loss_dict, losses, images, targets
        torch.cuda.empty_cache()
        loss_train.append(epoch_loss)
        # Evaluate
        epoch_train_acc, epoch_train_iou = 0, 0
        epoch_test_acc, epoch_test_iou = 0, 0
        model.eval()
        with torch.no_grad():
            for data_type in data_types:
                epoch_acc = 0.0
                epoch_iou = 0.0
                for images, targets in test_dataloader[data_type]:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    preds = model(images, targets)  # preds: [{boxes, labels, scores}]
                    batch_iou, batch_acc = calc_metrics(preds, targets)
                    epoch_acc += batch_acc
                    epoch_iou += batch_iou

                if data_type == 'train':
                    epoch_train_acc = 100 * (epoch_acc / datasets_sizes[data_type])
                    epoch_train_iou = epoch_iou / datasets_sizes[data_type]
                else:
                    epoch_test_acc = 100 * (epoch_acc / datasets_sizes[data_type])
                    epoch_test_iou = epoch_iou / datasets_sizes[data_type]
                scores[f'bbox_iou_{data_type}'].append(epoch_iou / datasets_sizes[data_type])
                scores[f'mask_accuracy_{data_type}'].append(100 * (epoch_acc / datasets_sizes[data_type]))
                del images, targets
                torch.cuda.empty_cache()

        print(f'Epoch {epoch + 1}/{epoch} done in {time.time() - epoch_start:.2f}s')
        print(f'\tTraining Loss {epoch_loss:.3f}')
        print(f'\tBounding Box IoU {epoch_train_iou:.3f}\tMask Accuracy {epoch_train_acc:.3f}')
        print(f'\tBounding Box IoU {epoch_test_iou:.3f}\tMask Accuracy {epoch_test_acc:.3f}\t(Test)')

        torch.save(model.state_dict(), f'masknet_{epoch}.torch')
    torch.save(model.state_dict(), 'masknet.torch')

    iou_plot = [(scores[f'bbox_iou_train'], 'Bounding Box IoU - Train'), (scores[f'bbox_iou_test'],\
                                                                          'Bounding Box IoU - Test')]
    mask_acc_plot = [(scores[f'mask_accuracy_test'], 'Mask Accuracy - Train'), (scores[f'mask_accuracy_test'],\
                                                                                'Mask Accuracy - Test)')]
    loss_plot = [(loss_train, 'Loss - Train')]
    fig = plt.figure()
    fig.suptitle('IoU Graph')
    for idx, (line, title) in enumerate(iou_plot):
        plt.subplot(1, 2, idx + 1)
        plt.plot(line)
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        plt.title(title)
    plt.subplots_adjust()
    plt.savefig('iou_fig.jpg')

    plt.clf()

    fig = plt.figure()
    fig.suptitle('Accuracy Graph')
    for idx, (line, title) in enumerate(mask_acc_plot):
        plt.subplot(1, 2, idx + 1)
        plt.plot(line)
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        plt.title(title)
    plt.subplots_adjust()
    plt.savefig('acc_mask_fig.jpg')

    plt.clf()

    fig = plt.figure()
    fig.suptitle('Loss Graph')
    for idx, (line, title) in enumerate(loss_plot):
        plt.plot(line)
        plt.xlabel('Epoch')
        plt.title(title)
    fig.tight_layout(pad=4.0)
    plt.subplots_adjust()
    plt.savefig('Loss_fig.jpg')


def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_types = ['train', 'test']
    # use our dataset and defined transformations
    train_dataset = dataset.MaskDataset('data/train', 'train')
    test_datasets = {
        data_type: dataset.MaskDataset(f'data/{data_type}','test') \
        for data_type in data_types}
    datasets_sizes = {data_type: len(test_datasets[data_type]) for data_type in data_types}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1,
                                                   collate_fn=collate_fn)
    test_dataloaders = {data_type : torch.utils.data.DataLoader(test_datasets[data_type], batch_size=32, shuffle=False,\
                                                               num_workers=1, collate_fn=collate_fn)\
                                                                for data_type in data_types}

    # get the model using our helper function
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=3,
                                                                 box_detections_per_img=1, min_size=224, max_size=224)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)

    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 5 epochs
    num_epochs = 10
    print("Strat training --> (The evaluation is happening  after each train epoch")
    train(model, optimizer, train_dataloader, test_dataloaders, data_types,datasets_sizes, device, num_epochs, lr_scheduler)
    print("That's it!")

if __name__ == '__main__':
    main()