import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import math
import numpy as np
import torch.nn.functional as F
import logging
import cv2

from torch.utils.data import DataLoader
from sample4geo.model import TimmModel
from sample4geo.dataset.university import get_transforms

DISTANCE_SCALE_M = 40.0
TILT_SCALE_DEG = 45.0
HEADING_SCALE_DEG = 180.0
XYZ_SCALE_M = 40.0


def angular_error_deg(pred, target):
    return torch.remainder(pred - target + 180.0, 360.0) - 180.0


def get_scheduler(optimizer, warmup_epochs, total_epochs, step_size, gamma):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            return gamma ** ((current_epoch - warmup_epochs) // step_size)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TourFrameImageDataset(Dataset):
    def __init__(self, json_dir, image_root, transforms=None):
        super().__init__()

        self.image_root = image_root
        self.transforms = transforms
        self.json_paths = [os.path.join(json_dir, g, f) for g in os.listdir(json_dir) for f in os.listdir(os.path.join(json_dir, g)) if f.endswith('.json')]

        self.samples = []

        for json_path in tqdm(self.json_paths, desc='Loading dataset'):
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_a_path = os.path.join(self.image_root, data['image_a'])
            image_b_path = os.path.join(self.image_root, data['image_b'])

            delta = data['delta']
            xyz = [float(delta[key]) for key in ('x_m', 'y_m', 'z_m')]
            tilt = float(data['tilt_num'])
            heading = float(data['heading_num'])

            self.samples.append({
                'image_a_path': image_a_path,
                'image_b_path': image_b_path,
                'xyz': xyz,
                'tilt': tilt,
                'heading': heading,
                'targets': [v / XYZ_SCALE_M for v in xyz] + [tilt / TILT_SCALE_DEG, heading / HEADING_SCALE_DEG],
                'json_path': json_path,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_a = cv2.imread(sample['image_a_path'])
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

        img_b = cv2.imread(sample['image_b_path'])
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img_a = self.transforms(image=img_a)['image']
            img_b = self.transforms(image=img_b)['image']

        targets = torch.tensor(sample['targets'], dtype=torch.float32)
        labels = torch.tensor(sample['xyz'] + [sample['tilt'], sample['heading']], dtype=torch.float32)
        return img_a, img_b, targets, labels, sample['json_path']


json_num = []

min_test_mse = float('inf')
min_distance_mae = float('inf')
min_tilt_mae = float('inf')
min_heading_mae = float('inf')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--lr_regressor', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=1e-10, type=float, dest='weight_decay')
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--warmup_epochs', default=1, type=int)
parser.add_argument('--image_root', type=str, required=True, help='Path to image root directory (tours)')
parser.add_argument('--train_dir', type=str, required=True, help='Path to training set JSON directory')
parser.add_argument('--test_dir', type=str, required=True, help='Path to test set JSON directory')
parser.add_argument('--output_file', type=str, default='test_results_e2e.log', help='Path to output file for test results')
parser.add_argument('--model_name', type=str, default='convnext_base.fb_in22k_ft_in1k_384')
parser.add_argument('--img_size', type=int, default=384)
parser.add_argument('--checkpoint', type=str, default='pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth',
                    help='Path to pretrained backbone checkpoint (empty string to skip)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights (only train regressor)')


class EndToEndModel(nn.Module):
    def __init__(self, model_name, pretrained=True, img_size=384):
        super(EndToEndModel, self).__init__()

        self.backbone = TimmModel(model_name, pretrained=pretrained, img_size=img_size)
        # convnext_base outputs 1024-dim features; two images concatenated = 2048
        self.regressor = nn.Linear(2048, 5)

    def forward(self, img_a, img_b):
        feat_a = self.backbone(img_a)
        feat_b = self.backbone(img_b)
        feat = torch.cat((feat_a, feat_b), dim=1)
        output = self.regressor(feat)
        return output


def validate(test_loader, model, device):
    model.eval()
    xyz_errors = [[], [], []]
    distance_errors = []
    tilt_errors = []
    heading_errors = []
    predictions = []
    labels_all = []

    with torch.no_grad():
        for img_a, img_b, _, labels, _ in tqdm(test_loader, desc='[Validate]'):
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(img_a, img_b)
            pred_xyz = output[:, :3] * XYZ_SCALE_M
            true_xyz = labels[:, :3]
            for i in range(3):
                xyz_errors[i].append(torch.abs(pred_xyz[:, i] - true_xyz[:, i]))
            pred_distance = torch.linalg.vector_norm(pred_xyz, dim=1)
            true_distance = torch.linalg.vector_norm(true_xyz, dim=1)
            pred_tilt = output[:, 3] * TILT_SCALE_DEG
            pred_heading = output[:, 4] * HEADING_SCALE_DEG
            distance_errors.append(torch.abs(pred_distance - true_distance))
            tilt_errors.append(torch.abs(pred_tilt - labels[:, 3]))
            heading_errors.append(torch.abs(angular_error_deg(pred_heading, labels[:, 4])))
            predictions.append(torch.cat((pred_xyz, pred_tilt[:, None], pred_heading[:, None]), dim=1).cpu())
            labels_all.append(labels.cpu())

    pred = torch.cat(predictions)
    truth = torch.cat(labels_all)
    distance_error = torch.abs(torch.linalg.vector_norm(pred[:, :3], dim=1) - torch.linalg.vector_norm(truth[:, :3], dim=1))
    success_count = int((distance_error <= 1.0).sum().item())
    success_rate = success_count / len(distance_error)
    metrics = tuple(torch.cat(values).mean() for values in (*xyz_errors, distance_errors, tilt_errors, heading_errors))
    return (*metrics, pred.numpy(), truth.numpy(), success_count, success_rate)

def train(train_loader, test_dataloader, model, criterion, optimizer, epoch, device, args, logger):
    global min_test_mse
    global train_losses
    global val_losses
    global pred_num
    global true_num
    global min_distance_mae
    global min_tilt_mae
    global min_heading_mae

    model.train()

    for img_a, img_b, targets, _, _ in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):

        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        output = model(img_a, img_b)
        loss = F.smooth_l1_loss(output, targets, beta=0.1, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x_mae, y_mae, z_mae, distance_mae, tilt_mae, heading_mae, pred, truth, success_count, success_rate = validate(test_dataloader, model, device)

    msg = 'Epoch {:d} | xyz MAE: {:.2f}/{:.2f}/{:.2f} m | Distance-from-xyz MAE: {:.2f} m | Tilt MAE: {:.2f} deg | Heading MAE: {:.2f} deg'.format(
        epoch, x_mae.item(), y_mae.item(), z_mae.item(), distance_mae.item(), tilt_mae.item(), heading_mae.item())
    logger.info(msg)

    metric = x_mae + y_mae + z_mae + distance_mae
    if metric < min_test_mse:
        min_test_mse = metric
        min_distance_mae = distance_mae
        min_tilt_mae = tilt_mae
        min_heading_mae = heading_mae

    best_msg = 'Best | Distance MAE: {:.2f} m | Tilt MAE: {:.2f} deg | Heading MAE: {:.2f} deg'.format(
        min_distance_mae.item(), min_tilt_mae.item(), min_heading_mae.item())
    logger.info(best_msg)

    output_base = os.path.splitext(args.output_file)[0]
    result = {
        'epoch': int(epoch + 1),
        'model': args.model_name,
        'train_rows': len(train_loader.dataset),
        'test_rows': len(test_dataloader.dataset),
        'MAE_x_m': float(x_mae.item()),
        'MAE_y_m': float(y_mae.item()),
        'MAE_z_m': float(z_mae.item()),
        'MAE_distance_from_xyz_m': float(distance_mae.item()),
        'MAE_tilt_deg': float(tilt_mae.item()),
        'MAE_heading_deg_circular': float(heading_mae.item()),
        'distance_success_threshold_m': 1.0,
        'distance_success_count': success_count,
        'distance_success_rate': float(success_rate),
        'distance_definition': 'MAE between norms of predicted and ground-truth xyz; no distance head',
    }
    with open(output_base + '.json', 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    torch.save({'model': model.state_dict(), 'metrics': result}, output_base + '.pt')
    np.savez_compressed(
        output_base + '_test_predictions.npz',
        pred_xyz_tilt_heading=pred,
        true_xyz_tilt_heading=truth,
    )

    return min_test_mse


def main():

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logger for dual output (console + file)
    logger = logging.getLogger('sample4geo_e2e')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(args.output_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Build end-to-end model
    model = EndToEndModel(model_name=args.model_name, pretrained=True, img_size=args.img_size)

    # Load pretrained backbone checkpoint if provided
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f'Loading backbone checkpoint: {args.checkpoint}')
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.backbone.load_state_dict(state_dict, strict=False)

    if args.freeze_backbone:
        logger.info('Freezing backbone parameters')
        for param in model.backbone.parameters():
            param.requires_grad = False

    model.to(device)

    # Image transforms
    img_size = (args.img_size, args.img_size)
    data_config = model.backbone.get_config()
    mean = data_config["mean"]
    std = data_config["std"]
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    train_dataset = TourFrameImageDataset(args.train_dir, args.image_root, transforms=train_drone_transforms)
    test_dataset = TourFrameImageDataset(args.test_dir, args.image_root, transforms=val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = None

    # Different learning rates for backbone and regressor
    param_groups = [
        {'params': list(model.regressor.parameters()), 'lr': args.lr_regressor}
    ]
    if not args.freeze_backbone:
        param_groups.append(
            {'params': list(model.backbone.parameters()), 'lr': args.lr_backbone}
        )

    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        step_size=30,
        gamma=0.1
    )

    for epoch in range(args.epochs):

        logger.info('Epoch: {}'.format(epoch))
        train(train_dataloader, test_dataloader, model, criterion, optimizer, epoch, device, args, logger)

        scheduler.step()

if __name__ == '__main__':
    main()
