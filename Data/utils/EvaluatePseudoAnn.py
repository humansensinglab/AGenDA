import argparse
from pycocotools.coco import COCO


def compute_detection_accuracy_iou50(gt_path, pred_path):
    # Load ground truth and prediction files
    coco_gt = COCO(gt_path)
    coco_pred = COCO(pred_path)

    TP = 0
    FP = 0
    total_gt = 0

    for img_id in coco_gt.getImgIds():
        gt_anns = coco_gt.imgToAnns.get(img_id, [])
        pred_anns = coco_pred.imgToAnns.get(img_id, [])
        gt_used = set()
        pred_used = set()

        # Match predictions to ground truth
        for pi, pred_ann in enumerate(pred_anns):
            pred_bbox = pred_ann['bbox']
            matched = False
            for gi, gt_ann in enumerate(gt_anns):
                if gi in gt_used:
                    continue
                gt_bbox = gt_ann['bbox']
                iou = compute_iou(gt_bbox, pred_bbox)
                if iou >= 0.5:
                    TP += 1
                    gt_used.add(gi)
                    pred_used.add(pi)
                    matched = True
                    break
            if not matched:
                FP += 1

        total_gt += len(gt_anns)

    FN = total_gt - TP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    
    print(f"Precision @ IoU 0.5: {precision:.4f}")
    print(f"Recall @ IoU 0.5: {recall:.4f}")
    return precision, recall


def compute_iou(boxA, boxB):
    # Format: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation.")
    parser.add_argument(
        "--ground_truth_file", type=str, help="ground truth pseudo annotation file path",
    )
    parser.add_argument(
        "--pseudo_pred_file", type=str, help="pseudo annotation save path",
    )

    args = parser.parse_args()
    ground_truth_file, pseudo_pred_file = args.ground_truth_file, args.pseudo_pred_file
    compute_detection_accuracy_iou50(ground_truth_file, pseudo_pred_file)