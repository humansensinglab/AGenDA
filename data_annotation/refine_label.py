import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import json
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import models
from torchmetrics import F1Score
import argparse


def construct_data(detection_results, pos_thresh, neg_thresh, hard_neg_thresh, bboxes_size_px=42.36, image_size=(112, 112), rgb_image_base_path=None):
    margin_px = bboxes_size_px/2-1

    coco_object_categories = [
        {'id': 1, 'name': 'small'}
    ]

    data_dict = dict(
        train_ann = [],
        test_ann = []
    )

    annotations_coco = dict(
        categories=coco_object_categories,
        images=[],
        annotations=[]
    )

    for i_im, detection_results_image in tqdm(enumerate(detection_results), total=len(detection_results)):
        heatmap_image_file_path = detection_results_image['img_path']
        image_file_name = os.path.basename(heatmap_image_file_path)
        
        rgb_image = Image.open(os.path.join(rgb_image_base_path,image_file_name)).convert("RGB")

        annotations_coco['images'].append(
            {
                'id': i_im,
                'file_name': image_file_name,
                'width': rgb_image.size[0],
                'height': rgb_image.size[1]
            }
        )
        
        pred_instances = detection_results_image['pred_instances']
        
        mask_instances_toExport = pred_instances['scores'] >= hard_neg_thresh
        
        scores = np.array(pred_instances['scores'][mask_instances_toExport])
        labels = np.array(pred_instances['labels'][mask_instances_toExport])
        bboxes = np.array(pred_instances['bboxes'][mask_instances_toExport,:])

        for i in range(bboxes.shape[0]):
            l, t, r, b = bboxes[i]
            s = scores[i]
            x_c_bbox = (l+r)/2
            y_c_bbox = (t+b)/2
            
            bbox_v_trimmed_edge = None
            bbox_h_trimmed_edge = None
            
            if x_c_bbox < margin_px:
                bbox_v_trimmed_edge = 'left'
            
            elif x_c_bbox > rgb_image.size[0] - margin_px:
                bbox_v_trimmed_edge = 'right'
        
            if y_c_bbox < margin_px:
                bbox_h_trimmed_edge = 'top'
            
            elif y_c_bbox > rgb_image.size[1] - margin_px:
                bbox_h_trimmed_edge = 'bottom'
        
            if bbox_v_trimmed_edge == 'left':
                r_full = r
                l_full = r - 42.36
        
            elif bbox_v_trimmed_edge == 'right':
                l_full = l
                r_full = l + 42.36
        
            else:
                l_full = l
                r_full = r
          
            if bbox_h_trimmed_edge == 'top':
                b_full = b
                t_full = b - 42.36
        
            elif bbox_h_trimmed_edge == 'bottom':
                t_full = t
                b_full = t + 42.36
        
            else:
                t_full = t
                b_full = b
        
            x_c_bbox_full = (l_full+r_full)/2
            y_c_bbox_full = (t_full+b_full)/2

            # Produce fake bbox annotations
            l = max(0, x_c_bbox_full-bboxes_size_px/2)
            t = max(0, y_c_bbox_full-bboxes_size_px/2)
            r = min(x_c_bbox_full+bboxes_size_px/2, image_size[0]-1)
            
            b = min(y_c_bbox_full+bboxes_size_px/2, image_size[1]-1)
            w_bbox = r - l
            h_bbox = b - t

            crop_img = rgb_image.crop((l,t,r,b))

            if i ==0 or s >= pos_thresh: # top one and most confident samples
                data_dict['train_ann'].append(
                    {
                        'img': crop_img,
                        'id': len(data_dict['train_ann']),
                        'label': 1,
                    }
                )
                annotations_coco['annotations'].append(
                    {
                        'iscrowd': 0,
                        'category_id': coco_object_categories[labels[i]]['id'],
                        'image_id': i_im, 
                        'bbox': [l, t, w_bbox, h_bbox],
                        'area': w_bbox*h_bbox,
                        'label': 1,
                    }
                )

            elif s < neg_thresh:
                data_dict['train_ann'].append(
                    {
                        'img': crop_img,
                        'id': len(data_dict['train_ann']),
                        'label': 0
                    }
                )

            else:
                data_dict['test_ann'].append(
                    {
                        'iscrowd': 0,
                        'category_id': coco_object_categories[labels[i]]['id'], 
                        'image_id': i_im, 
                        'bbox': [l, t, w_bbox, h_bbox],
                        'area': w_bbox*h_bbox,
                        'img': crop_img,
                        'id': len(data_dict['test_ann']),
                        'label': -1
                    }
                )
    
    return data_dict, annotations_coco


class CustomDataset(nn.Module):
    def __init__(self, data_dict, transform, split="train"):
        super().__init__()
        if split=="train":
            self.data = data_dict['train_ann']
        else:
            self.data = data_dict['test_ann']
        self.T = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ann = self.data[index]
        img = ann['img']
        label = ann['label']
        ids = ann['id']
        img_t = self.T(img)
        # stacked = torch.concatenate([img_t, heatmap_t])
        return {
            # 'images':stacked,
            'images':img_t,
            'labels':label,
            'ids':ids
        }


def train(model, criterion, optimizer, pbar, epoch, device):
    model.train()

    for i, batch in enumerate(pbar):
        images, labels = batch['images'].to(device), batch['labels'].to(dtype=torch.float32, device=device)
        labels = labels.unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item():.4f}")


def evaluate(model, dataloader):
    model.eval()

    pred_all = []
    label_all = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            images, labels = batch['images'].to(device), batch['labels'].to(device)
            labels = labels.unsqueeze(1)
            outputs = model(images)          
            pred = outputs > 0
            pred_all.append(pred)
            label_all.append(labels)

    pred_all = torch.concatenate(pred_all)
    label_all = torch.concatenate(label_all)
    return pred_all, label_all


def test(model, dataloader):
    model.eval()
    pos_ids = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, ids = batch['images'].to(device), batch['ids'].to(device)
            outputs = model(images)
           
            pred = (outputs > 0).squeeze(1)
            pos_ids += ids[pred].detach().cpu().tolist()

    return pos_ids



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--prediction_pkl", type=str, help="prediction file path"
    )
    parser.add_argument(
        "--synthetic_image_base_path", type=str, help="image path"
    )
    parser.add_argument(
        "--json_save_path", type=str, help="prediction json save path"
    )
    parser.add_argument(
        "--checkpoint_save_path", type=str, help="classifier checkpoint save path"
    )
    parser.add_argument(
        "--pos_thresh", type=float, default=0.75, help="positive sample threshold"
    )
    parser.add_argument(
        "--neg_thresh", type=float, default=0.35, help="negative sample threshold"
    )
    parser.add_argument(
        "--hard_neg_thresh", type=float, default=0.05, help="filter out extremely low conf bbox"
    )
    parser.add_argument(
        "--num_classes", type=int, default=1
    )
    parser.add_argument(
        "--num_epochs", type=int, default=80
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=256
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=512
    )
    parser.add_argument(
        "--lr", type=float, default=4e-4
    )
    args = parser.parse_args()

    os.makedirs(args.checkpoint_save_path,exist_ok=True)
    with open(args.prediction_pkl, 'rb') as f:
        detection_results = pickle.load(f)

    best_acc = 0
    best_f1 = 0

    data_dict, annotations_coco = construct_data(detection_results, args.pos_thresh, args.neg_thresh, args.hard_neg_thresh, rgb_image_base_path=args.synthetic_image_base_path)
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(data_dict, train_transform)
    val_dataset = CustomDataset(data_dict, test_transform)
    test_dataset = CustomDataset(data_dict, test_transform, split="test")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=4
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4
    )

    model = models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d(input_channel, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model = model.to(device)
    # model.load_state_dict(torch.load(os.path.join(checkpoint_save_path,f"resnet_best_f1.pth"),map_location="cpu"))

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    f1_metric = F1Score('multiclass',num_classes=args.num_classes+1, average="macro").to(device)

    for epoch in range(args.num_epochs):
        pbar = tqdm(train_dataloader) 
        train(model, criterion, optimizer, pbar, epoch, device)
        pred_all, label_all = evaluate(model, val_dataloader)
        accuracy = torch.sum(pred_all == label_all).item() / label_all.shape[0]
        f1 = f1_metric(pred_all, label_all)
        print(f"Epoch {epoch}: Train Accuracy: {accuracy:.4f}, Train f1: {f1:.4f}")
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(),os.path.join(args.checkpoint_save_path,f"resnet_best_accuracy.pth"))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(),os.path.join(args.checkpoint_save_path,f"resnet_best_f1.pth"))

    # testing
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_save_path,f"resnet_best_f1.pth"),map_location="cpu"))
    model = model.to(device)   
    pos_ids = test(model,test_dataloader)

    for pos_id in tqdm(pos_ids):
        ann_item = next((item for item in data_dict['test_ann'] if item["id"] == pos_id))
        annotations_coco['annotations'].append({
            'iscrowd': ann_item['iscrowd'],
            'category_id': ann_item['category_id'],
            'image_id': ann_item['image_id'], 
            'bbox': ann_item['bbox'],
            'area': ann_item['area'],
            'label':-1,
        })
    
    annotations_coco['annotations'] = sorted(annotations_coco['annotations'], key=lambda x: x["image_id"])
    item_id = 0
    for ann in annotations_coco['annotations']:
        ann['id'] = item_id
        item_id +=1
    
    with open(args.json_save_path, "w") as f:
        json.dump(annotations_coco,f)

                        
