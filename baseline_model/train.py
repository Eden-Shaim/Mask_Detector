import dataset
import mask_detector
import torch
import torch.utils.data as data
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from functools import partial
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
#
print(torch.__version__)

def load_data(train_path, test_path):
    train_dataset = dataset.MaskDataset(train_path)
    test_dataset = dataset.MaskDataset(test_path)
    return train_dataset, test_dataset


def collate(batch):
    images, bounding_boxes, masks = [], [], []
    for image, bbox, mask in batch:
        images.append(image)
        bounding_boxes.append(bbox)
        masks.append(mask)
    return torch.stack(images), torch.stack(bounding_boxes), torch.tensor(masks, dtype=torch.long)


def intersection_over_union(bbox1, bbox2):
    tl1, tl2 = bbox1[:, :2], bbox2[:, :2]
    br1, br2 = tl1 + bbox1[:, 2:], tl2 + bbox2[:, 2:]
    area1 = bbox1[:, 2] * bbox1[:, 3]
    area2 = bbox2[:, 2] * bbox2[:, 3]
    min_b = torch.max(tl1, tl2)
    max_b = torch.min(br1, br2)
    overlap = torch.clamp(max_b - min_b, min=0)
    intersection = overlap[:, 0] * overlap[:, 1]
    union = area1 + area2 - intersection
    return intersection / union


def test(model, dataloader):
    l1_loss = torch.nn.L1Loss()
    bce_loss = torch.nn.BCELoss()
    bbox_epoch_loss, mask_epoch_loss = 0, 0
    bbox_epoch_iou, mask_epoch_accuracy = 0, 0
    with torch.no_grad():
        for image, true_bbox, true_mask in dataloader:
            image, true_bbox, true_mask = image.cuda(), true_bbox.float().cuda(), true_mask.cuda()
            bbox, mask = model(image)
            bbox_batch_loss = l1_loss(bbox, true_bbox)
            mask_batch_loss = bce_loss(mask, true_mask.float())
            # loss_test = bbox_batch_loss_test + mask_batch_loss_test
            bbox_epoch_loss += bbox_batch_loss.item() / len(dataloader)
            mask_epoch_loss += mask_batch_loss.item() / len(dataloader)
            bbox_epoch_iou += intersection_over_union(bbox, true_bbox).sum().item() / len(dataloader.dataset)
            mask: torch.Tensor
            mask_epoch_accuracy += mask.ge(0.5).eq(true_mask).float().sum().item() / len(dataloader.dataset)
    return bbox_epoch_loss, mask_epoch_loss, bbox_epoch_iou, mask_epoch_accuracy


def train(data_dir, n_epoch=40, batch_size=32, lr=0.02):
    # train_dataset, test_dataset = load_data(Path(data_dir + 'train'), Path(data_dir + 'test'))
    train_dataset = dataset.MaskDataset(Path('data/train'), 'tarin')
    test_dataset = dataset.MaskDataset(Path('data/test'), 'test')
    data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
    print(data_loader)
    model = mask_detector.MaskDetector()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    l1_loss = torch.nn.L1Loss()
    bce_loss = torch.nn.BCELoss()
    criterion = nn.CrossEntropyLoss().to(device)
    bbox_loss, mask_loss = [], []
    bbox_loss_test, mask_loss_test = [], []
    bbox_iou, mask_accuracy = [], []
    bbox_test_iou, mask_test_accuracy = [], []
    for epoch in range(n_epoch):
        epoch_start = time.time()
        bbox_epoch_loss, mask_epoch_loss = 0, 0

        bbox_epoch_iou, mask_epoch_accuracy = 0, 0
        for image, true_bbox, true_mask in data_loader:
            image, true_bbox, true_mask = image.cuda(), true_bbox.float().cuda(), true_mask.cuda()
            bbox, mask = model(image)
            bbox_batch_loss = l1_loss(bbox, true_bbox)
            # mask_batch_loss = bce_loss(mask, true_mask.float())
            mask_batch_loss = criterion(mask, true_mask)
            loss = bbox_batch_loss + mask_batch_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bbox_epoch_loss += bbox_batch_loss.item() / len(data_loader)
            mask_epoch_loss += mask_batch_loss.item() / len(data_loader)

            _, proper_mask_pred = torch.max(mask, 1)
            bbox_epoch_iou += intersection_over_union(bbox, true_bbox).sum().item() / len(data_loader.dataset)
            # mask_epoch_accuracy += mask.ge(0.5).eq(true_mask).float().sum().item() / len(data_loader.dataset)
            mask_epoch_accuracy += (proper_mask_pred == true_mask).float().sum().item() / len(data_loader.dataset)

        bbox_loss.append(bbox_epoch_loss)
        mask_loss.append(mask_epoch_loss)
        bbox_iou.append(bbox_epoch_iou)
        mask_accuracy.append(mask_epoch_accuracy)
        print(f'Epoch {epoch + 1}/{n_epoch} done in {time.time() - epoch_start:.2f}s')
        print(f'\tBounding Box Loss {bbox_epoch_loss:.3f}\tMask Loss {mask_epoch_loss:.3f}')
        print(f'\tBounding Box IoU {bbox_epoch_iou:.3f}\tMask Accuracy {mask_epoch_accuracy:.3f}')
        bbox_epoch_loss_test, mask_epoch_loss_test, bbox_epoch_iou_test, mask_epoch_accuracy_test = test(model, test_loader)
        bbox_loss_test.append(bbox_epoch_loss_test)
        mask_loss_test.append(mask_epoch_loss_test)
        bbox_test_iou.append(bbox_epoch_iou_test)
        mask_test_accuracy.append(mask_epoch_accuracy_test)

        print(f'\tBounding Box IoU {bbox_epoch_iou_test:.3f}\tMask Accuracy {mask_epoch_accuracy_test:.3f}\t(Test)')
    torch.save(model.state_dict(), 'masknet.torch')
    iou_plot = [(bbox_iou, 'Bounding Box IoU - Train'), (bbox_test_iou, 'Bounding Box IoU - Test')]
    mask_acc_plot = [(mask_accuracy, 'Mask Accuracy - Train'), (mask_test_accuracy, 'Mask Accuracy - Test)')]
    loss_plots = [(bbox_loss, 'Bounding Box Loss - Train'), (mask_loss, 'Mask Loss - Train'),
                  (bbox_loss_test, 'Bounding Box Loss - Test'), (mask_loss_test, 'Mask Loss - Test')]
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
    for idx, (line, title) in enumerate(loss_plots):
        plt.subplot(2, 2, idx + 1)
        plt.plot(line)
        plt.xlabel('Epoch')
        plt.title(title)
    fig.tight_layout(pad=4.0)
    plt.subplots_adjust()
    plt.savefig('Loss_fig.jpg')


def train_for_hyperparameter_tuning(config, data_dir=None, checkpoint_dir=None, n_epoch=40):
    train_dataset, test_dataset = load_data(Path(data_dir + 'train'), Path(data_dir + 'test'))
    data_loader = data.DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True, collate_fn=collate)
    test_loader = data.DataLoader(test_dataset, batch_size=int(config["batch_size"]), collate_fn=collate)
    model = mask_detector.MaskDetector()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    l1_loss = torch.nn.L1Loss()
    bce_loss = torch.nn.BCELoss()

    bbox_loss, mask_loss = [], []
    bbox_loss_test, mask_loss_test = [], []
    bbox_iou, mask_accuracy = [], []
    bbox_test_iou, mask_test_accuracy = [], []
    for epoch in range(n_epoch):
        epoch_start = time.time()
        bbox_epoch_loss, mask_epoch_loss = 0, 0

        bbox_epoch_iou, mask_epoch_accuracy = 0, 0
        for image, true_bbox, true_mask in data_loader:
            image, true_bbox, true_mask = image.cuda(), true_bbox.float().cuda(), true_mask.cuda()
            bbox, mask = model(image)
            bbox_batch_loss = l1_loss(bbox, true_bbox)
            mask_batch_loss = bce_loss(mask, true_mask.float())
            loss = bbox_batch_loss + mask_batch_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bbox_epoch_loss += bbox_batch_loss.item() / len(data_loader)
            mask_epoch_loss += mask_batch_loss.item() / len(data_loader)
            bbox_epoch_iou += intersection_over_union(bbox, true_bbox).sum().item() / len(data_loader.dataset)
            mask_epoch_accuracy += mask.ge(0.5).eq(true_mask).float().sum().item() / len(data_loader.dataset)

        bbox_loss.append(bbox_epoch_loss)
        mask_loss.append(mask_epoch_loss)
        bbox_iou.append(bbox_epoch_iou)
        mask_accuracy.append(mask_epoch_accuracy)
        print(f'Epoch {epoch + 1}/{n_epoch} done in {time.time() - epoch_start:.2f}s')
        print(f'\tBounding Box Loss {bbox_epoch_loss:.3f}\tMask Loss {mask_epoch_loss:.3f}')
        print(f'\tBounding Box IoU {bbox_epoch_iou:.3f}\tMask Accuracy {mask_epoch_accuracy:.3f}')
        bbox_epoch_loss_test, mask_epoch_loss_test, bbox_epoch_iou_test, mask_epoch_accuracy_test = test(model, test_loader)
        bbox_loss_test.append(bbox_epoch_loss_test)
        mask_loss_test.append(mask_epoch_loss_test)
        bbox_test_iou.append(bbox_epoch_iou_test)
        mask_test_accuracy.append(mask_epoch_accuracy_test)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=bbox_epoch_loss_test, accuracy=bbox_epoch_iou)
        tune.report(loss=mask_epoch_loss_test, accuracy=mask_epoch_accuracy)

        print(f'\tBounding Box IoU {bbox_epoch_iou_test:.3f}\tMask Accuracy {mask_epoch_accuracy_test:.3f}\t(Test)')
    torch.save(model.state_dict(), 'masknet.torch')
    iou_plot = [(bbox_iou, 'Bounding Box IoU - Train'), (bbox_test_iou, 'Bounding Box IoU - Test')]
    mask_acc_plot = [(mask_accuracy, 'Mask Accuracy - Train'), (mask_test_accuracy, 'Mask Accuracy - Test)')]
    loss_plots = [(bbox_loss, 'Bounding Box Loss - Train'), (mask_loss, 'Mask Loss - Train'),
                  (bbox_loss_test, 'Bounding Box Loss - Test'), (mask_loss_test, 'Mask Loss - Test')]
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
    for idx, (line, title) in enumerate(loss_plots):
        plt.subplot(2, 2, idx + 1)
        plt.plot(line)
        plt.xlabel('Epoch')
        plt.title(title)
    plt.subplots_adjust()
    plt.savefig('Loss_fig.jpg')


def main_for_tuning(num_samples=10, max_num_epochs=50, gpus_per_trial=2):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train, data_dir='data/'),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    best_trained_model = mask_detector.MaskDetector()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)


if __name__ == '__main__':
    train('data/')
    # main_for_tuning(num_samples=10, max_num_epochs=50, gpus_per_trial=2)