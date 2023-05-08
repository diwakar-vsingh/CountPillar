from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def get_img_and_mask(
    pill_mask_paths: Tuple[Path, Path], thresh: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the image and mask from the pill mask paths.

    Args:
        pill_mask_paths: The paths to the pill image and mask.
        thresh: The threshold at which to binarize the mask.

    Returns:
        img: The pill image.
        mask: The pill mask.
    """
    img_path, mask_path = pill_mask_paths
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask[mask <= thresh] = 0
    mask[mask > thresh] = 1

    return img, mask


def create_yolo_annotations(
    mask_comp: np.ndarray, labels_comp: List[int]
) -> List[List[float]]:
    """Create YOLO annotations from a composition mask.

    Args:
        mask_comp: The composition mask.
        labels_comp: The labels of the composition.

    Returns:
        annotations_yolo: The YOLO annotations.
    """
    comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]

    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
    masks = mask_comp == obj_ids[:, None, None]

    annotations_yolo: List[List[float]] = []
    for i in range(len(labels_comp)):
        pos = np.where(masks[i])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        ymin, ymax = np.min(pos[0]), np.max(pos[0])

        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        w, h = xmax - xmin, ymax - ymin

        annotations_yolo.append(
            [
                labels_comp[i] - 1,
                round(xc / comp_w, 5),
                round(yc / comp_h, 5),
                round(w / comp_w, 5),
                round(h / comp_h, 5),
            ]
        )

    return annotations_yolo