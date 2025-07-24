import os
import json
import argparse
import numpy as np
from fuzzywuzzy import fuzz
from collections import defaultdict
from typing import Dict, List, Tuple

def points_to_rectangles(coordinates: List) -> List[Tuple[int, int, int, int]]:
    """Convert nested list of coordinate points into rectangles."""
    if not coordinates:
        return []
    
    rectangles = []
    for coord_group in coordinates:
        if not isinstance(coord_group, list) or len(coord_group) < 4:
            continue
        points = np.array([[c['x'], c['y']] for c in coord_group if isinstance(c, dict)])
        if len(points) < 4:
            continue
        # Remove duplicates
        points = np.unique(points, axis=0)
        # Compute bounding box for the group
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        if x_max > x_min and y_max > y_min:
            rectangles.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    
    # Merge overlapping rectangles
    merged = []
    while rectangles:
        rect = rectangles.pop(0)
        x1, y1, x2, y2 = rect
        overlapping = []
        for other in rectangles:
            ox1, oy1, ox2, oy2 = other
            if not (x2 < ox1 or ox2 < x1 or y2 < oy1 or oy2 < y1):
                overlapping.append(other)
                x1 = min(x1, ox1)
                y1 = min(y1, oy1)
                x2 = max(x2, ox2)
                y2 = max(y2, oy2)
        for ov in overlapping:
            rectangles.remove(ov)
        if x2 > x1 and y2 > y1:
            merged.append((x1, y1, x2, y2))
    
    return merged

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_g, y1_g, x2_g, y2_g = box2
    if x1 >= x2 or y1 >= y2 or x1_g >= x2_g or y1_g >= y2_g:
        print(f"Invalid boxes: Ext {box1}, GT {box2}")
        return 0.0
    xi1 = max(x1, x1_g)
    yi1 = max(y1, y1_g)
    xi2 = min(x2, x2_g)
    yi2 = min(y2, y2_g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_g - x1_g) * (y2_g - y1_g)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    if iou > 0.999:
        print(f"Warning: Near-perfect IoU {iou:.4f} for boxes Ext {box1}, GT {box2}")
    return iou

def evaluate_pii_file(extracted_path: str, ground_truth_path: str) -> Dict:
    """Evaluate extracted PII against ground truth, emphasizing coordinates."""
    try:
        with open(extracted_path, 'r', encoding='utf-8') as f:
            extracted = json.load(f)
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    except Exception as e:
        print(f"Error loading files: {extracted_path}, {ground_truth_path}: {str(e)}")
        return {}

    fields = ["Name", "Father's Name", "Mother's Name", "Husband's Name", "Phone Number",
              "Date of Birth", "Aadhaar ID", "PAN ID", "Passport ID", "Driving License ID",
              "Voter ID", "Address", "ZIP Code", "Bill Number"]
    
    metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0, 'iou_count': 0, 'coord_valid': 0, 'coord_fp': 0, 'coord_fn': 0})
    
    for field in fields:
        ext_data = extracted.get(field, {"value": None, "coordinates": []})
        gt_data = ground_truth.get(field, {"value": None, "coordinates": []})
        
        ext_value = ext_data.get("value", None)
        gt_value = gt_data.get("value", None)
        ext_coords = ext_data.get("coordinates", [])
        gt_coords = gt_data.get("coordinates", [])
        
        # Value comparison (no normalization)
        if ext_value is not None and gt_value is not None:
            is_match = fuzz.ratio(str(ext_value).lower(), str(gt_value).lower()) >= 80 if field in ["Name", "Father's Name", "Mother's Name", "Husband's Name", "Address"] else str(ext_value) == str(gt_value)
            if is_match:
                metrics[field]['tp'] += 1
                print(f"Match for {field}: Extracted '{ext_value}' == Ground Truth '{gt_value}'")
            else:
                metrics[field]['fp'] += 1
                metrics[field]['fn'] += 1
                print(f"Mismatch for {field}: Extracted '{ext_value}' vs. Ground Truth '{gt_value}'")
            
            # Coordinate comparison
            if ext_coords and gt_coords:
                ext_boxes = points_to_rectangles(ext_coords)
                gt_boxes = points_to_rectangles(gt_coords)
                matched_gt = set()  # Track matched ground truth boxes
                matched_ext = set()  # Track matched extracted boxes
                has_valid_coord = False
                
                # Try to match each ground truth box to an extracted box
                for j, gt_box in enumerate(gt_boxes):
                    best_iou = 0.0
                    best_ext_idx = None
                    for i, ext_box in enumerate(ext_boxes):
                        if i in matched_ext:
                            continue
                        iou = compute_iou(ext_box, gt_box)
                        print(f"Comparing {field} rectangles: Ext {ext_box}, GT {gt_box}, IoU: {iou:.2f}")
                        if iou >= 0.5 and iou > best_iou:
                            best_iou = iou
                            best_ext_idx = i
                    if best_ext_idx is not None:
                        metrics[field]['iou_sum'] += best_iou
                        metrics[field]['iou_count'] += 1
                        matched_gt.add(j)
                        matched_ext.add(best_ext_idx)
                        has_valid_coord = True
                        print(f"IoU for {field}: {best_iou:.2f} (Valid for masking)")
                
                # Handle unmatched ground truth coordinates (missed coordinates)
                unmatched_gt_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]
                if unmatched_gt_indices:
                    unmatched_gt_count = len(unmatched_gt_indices)
                    metrics[field]['coord_fn'] += unmatched_gt_count
                    unmatched_gt_boxes = [gt_boxes[i] for i in unmatched_gt_indices]
                    print(f"Missed {unmatched_gt_count} ground truth coordinates for {field}: {unmatched_gt_boxes}")
                
                # Handle unmatched extracted coordinates (extra coordinates)
                unmatched_ext_indices = [i for i in range(len(ext_boxes)) if i not in matched_ext]
                if unmatched_ext_indices:
                    unmatched_ext_count = len(unmatched_ext_indices)
                    metrics[field]['coord_fp'] += unmatched_ext_count
                    unmatched_ext_boxes = [ext_boxes[i] for i in unmatched_ext_indices]
                    print(f"Extra {unmatched_ext_count} coordinates for {field} in extracted data: {unmatched_ext_boxes}")
                
                if has_valid_coord:
                    metrics[field]['coord_valid'] = 1  # Count once per field if at least one valid coordinate match
            elif ext_coords and not gt_coords:
                ext_boxes = points_to_rectangles(ext_coords)
                metrics[field]['coord_fp'] += len(ext_boxes)
                print(f"Extra {len(ext_boxes)} coordinates for {field} in extracted data: {ext_coords}")
            elif gt_coords and not ext_coords:
                gt_boxes = points_to_rectangles(gt_coords)
                metrics[field]['coord_fn'] += len(gt_boxes)
                print(f"Missed {len(gt_boxes)} ground truth coordinates for {field}: {gt_coords}")
        elif ext_value is not None and gt_value is None:
            metrics[field]['fp'] += 1
            ext_boxes = points_to_rectangles(ext_coords)
            metrics[field]['coord_fp'] += len(ext_boxes)
            print(f"False Positive for {field}: Extracted '{ext_value}', Ground Truth: None")
            if ext_boxes:
                print(f"Extra {len(ext_boxes)} coordinates for {field} in extracted data: {ext_coords}")
        elif gt_value is not None and ext_value is None:
            metrics[field]['fn'] += 1
            gt_boxes = points_to_rectangles(gt_coords)
            metrics[field]['coord_fn'] += len(gt_boxes)
            print(f"False Negative for {field}: Ground Truth '{gt_value}', Extracted: None")
            if gt_boxes:
                print(f"Missed {len(gt_boxes)} ground truth coordinates for {field}: {gt_coords}")
    
    return metrics

def compute_metrics(metrics: Dict) -> Dict:
    """Compute precision, recall, F1-score, average IoU, and coordinate validity rate."""
    result = {}
    total_tp, total_fp, total_fn, total_coord_valid, total_coord_fp, total_coord_fn = 0, 0, 0, 0, 0, 0
    for field, counts in metrics.items():
        tp = counts['tp']
        fp = counts['fp'] + counts['coord_fp']  # Include extra coordinates as false positives
        fn = counts['fn'] + counts['coord_fn']  # Include missed coordinates as false negatives
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        avg_iou = counts['iou_sum'] / counts['iou_count'] if counts['iou_count'] > 0 else 0.0
        coord_valid_rate = counts['coord_valid'] / (tp + counts['fp']) if tp + counts['fp'] > 0 else 0.0
        result[field] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_iou': avg_iou,
            'coord_valid_rate': coord_valid_rate,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'coord_fp': counts['coord_fp'],
            'coord_fn': counts['coord_fn'],
            'iou_count': counts['iou_count']
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_coord_valid += counts['coord_valid']
        total_coord_fp += counts['coord_fp']
        total_coord_fn += counts['coord_fn']
    
    # Compute overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0.0
    overall_accuracy = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn > 0 else 0.0
    overall_coord_valid_rate = total_coord_valid / (total_tp + total_fp - total_coord_fn) if total_tp + total_fp - total_coord_fn > 0 else 0.0
    result['Overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'accuracy': overall_accuracy,
        'coord_valid_rate': overall_coord_valid_rate,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'coord_fp': total_coord_fp,
        'coord_fn': total_coord_fn
    }
    
    return result

def main():
    """Evaluate PII extraction against ground truth, focusing on coordinates and text accuracy."""
    parser = argparse.ArgumentParser(description="Evaluate PII extraction performance.")
    parser.add_argument('--extracted_dir', type=str, required=True, help="Directory containing extracted PII JSON files")
    parser.add_argument('--ground_truth_dir', type=str, required=True, help="Directory containing ground truth JSON files")
    parser.add_argument('--output_metrics', type=str, required=True, help="Path to save output metrics JSON file")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)

    # Get ground truth files
    gt_files = [f for f in os.listdir(args.ground_truth_dir) if f.lower().endswith('.json')]
    print(f"Found {len(gt_files)} ground truth files.")

    aggregated_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0, 'iou_count': 0, 'coord_valid': 0, 'coord_fp': 0, 'coord_fn': 0})

    for gt_file in gt_files:
        # Match extracted files
        base_name = gt_file.replace('_pii.json', '.json') if '_pii.json' in gt_file else gt_file
        extracted_path = os.path.join(args.extracted_dir, base_name)
        ground_truth_path = os.path.join(args.ground_truth_dir, gt_file)

        if not os.path.exists(extracted_path):
            print(f"Warning: Extracted file '{extracted_path}' not found. Skipping.")
            continue

        print(f"\nEvaluating: {gt_file}")
        file_metrics = evaluate_pii_file(extracted_path, ground_truth_path)

        for field, counts in file_metrics.items():
            for key in ['tp', 'fp', 'fn', 'iou_sum', 'iou_count', 'coord_valid', 'coord_fp', 'coord_fn']:
                aggregated_metrics[field][key] += counts[key]

    # Compute and save metrics
    final_metrics = compute_metrics(aggregated_metrics)
    with open(args.output_metrics, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to: {args.output_metrics}")

    # Print overall summary
    print("\nOverall Evaluation Summary:")
    overall = final_metrics['Overall']
    print(f"  Precision: {overall['precision']:.2f}")
    print(f"  Recall: {overall['recall']:.2f}")
    print(f"  F1-Score: {overall['f1_score']:.2f}")
    print(f"  Accuracy: {overall['accuracy']:.2f}")
    print(f"  Coordinate Validity Rate: {overall['coord_valid_rate']:.2f} (valid coordinates)")
    print(f"  TP: {overall['tp']}, FP: {overall['fp']}, FN: {overall['fn']}")
    print(f"  Coordinate FP: {overall['coord_fp']}, Coordinate FN: {overall['coord_fn']}")

    # Print per-field summary
    print("\nPer-Field Evaluation Summary:")
    for field in sorted(final_metrics.keys()):
        if field == 'Overall':
            continue
        metrics = final_metrics[field]
        print(f"{field}:")
        print(f"  Precision: {metrics['precision']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        print(f"  F1-Score: {metrics['f1_score']:.2f}")
        print(f"  Avg IoU: {metrics['avg_iou']:.2f} (based on {metrics['iou_count']} matches)")
        print(f"  Coordinate Validity Rate: {metrics['coord_valid_rate']:.2f} (valid coordinates)")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
        print(f"  Coordinate FP: {metrics['coord_fp']}, Coordinate FN: {metrics['coord_fn']}")

if __name__ == "__main__":
    main()
