"""
M5 K‑means++ anchor clustering (Experimental Prototype)
Generate improved anchor priors for nectar‑plant flower dataset.
Usage:
    python utils/kmeans_plus_plus_anchor.py --label-dir ./data/wubeizi/train/labels --n-anchor 9
Output: anchor list, manually copy to your model yaml 'anchors:' entry.
Note: This script runs offline before training, NOT invoked during training loop.
After pasting anchors into yaml, autoanchor.py will validate anchor quality automatically in train.py.
"""
import numpy as np
import os
import argparse


def iou(box, clusters):
    """
    box: (w, h); clusters: shape(N,2)
    calculate IoU between single box and each cluster
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has zero‑area, check label files")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def kmeans_plus_plus(boxes, n_clusters):
    """
    K‑means++ init + standard k‑means iteration
    boxes: array(N,2), each row is (w,h) normalized from label txt
    n_clusters: number of anchor clusters
    return cluster centroids (n_clusters,2)
    """
    n = boxes.shape[0]
    centroids = []
    # k‑means++ initial centroid selection
    idx = np.random.randint(0, n)
    centroids.append(boxes[idx].copy())

    for _ in range(1, n_clusters):
        dist_list = []
        for b in boxes:
            ious = iou(b, np.array(centroids))
            dist = 1.0 - np.max(ious)
            dist_list.append(dist)
        dist_list = np.array(dist_list)
        prob = dist_list / np.sum(dist_list)
        select_idx = np.random.choice(np.arange(n), p=prob)
        centroids.append(boxes[select_idx].copy())

    centroids = np.array(centroids)
    # standard k‑means loop
    for _ in range(300):
        assignments = [[] for _ in range(n_clusters)]
        for b in boxes:
            ious = iou(b, centroids)
            assignments[np.argmax(ious)].append(b)
        new_centroids = []
        for group in assignments:
            group_arr = np.array(group)
            new_centroids.append(np.median(group_arr, axis=0))
        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids


def load_label_boxes(label_dir):
    """load w h from yolo‑format label txt files"""
    boxes = []
    for fname in os.listdir(label_dir):
        if not fname.lower().endswith(".txt"):
            continue
        fpath = os.path.join(label_dir, fname)
        with open(fpath, "r", encoding="utf‑8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = list(map(float, line.split()))
                _, _, _, w, h = parts[:5]
                boxes.append([w, h])
    return np.array(boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label‑dir", type=str, required=True, help="path to train labels folder (*.txt yolo labels)")
    parser.add_argument("--n‑anchor", type=int, default=9, help="number of anchors, default=9 for yolov5s 3 detection layers")
    opt = parser.parse_args()

    box_data = load_label_boxes(opt.label_dir)
    print(f"Loaded {box_data.shape[0]} object boxes from labels")
    out_clusters = kmeans_plus_plus(box_data, opt.n_anchor)
    # convert to flattened integer‑style anchor for yaml (multiply by 640 input image size)
    img_size = 640
    anchors_scaled = (out_clusters * img_size).astype(int).reshape(-1).tolist()
    print("\n==== Generated anchors (copy‑paste to your model yaml anchors field): ====")
    print(anchors_scaled)
    print("\nExample yaml snippet:")
    print("anchors:\n  -", anchors_scaled[:6])
    print("  -", anchors_scaled[6:12])
    print("  -", anchors_scaled[12:])
    print("\nAfter writing anchors in yaml, train.py will use autoanchor.py to validate anchors automatically.")
