from pathlib import Path
from typing import List, Tuple

import numpy as np

from countpillar.io_utils import get_img_and_mask
from countpillar.object_overlay import add_pill_on_bg, verify_overlap
from countpillar.transform import resize_and_transform_pill


def create_pill_comp(
    bg_img: np.ndarray,
    pill_mask_paths: List[Tuple[Path, Path]],
    min_pills: int = 5,
    max_pills: int = 15,
    max_overlap: float = 0.2,
    max_attempts: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Create a composition of pills on a background image.

    Args:
        bg_img: The background image.
        pill_mask_paths: List of tuples of pill image and mask paths.
        min_pills: The minimum number of pills to compose.
        max_pills: The maximum number of pills to compose.
        max_overlap: The maximum allowed overlap between pills.
        max_attempts: The maximum number of attempts to compose a pill.

    Returns:
        bg_img: The background image with pills.
        composition_mask: The mask of the composition.
        pill_areas: List of areas of the pills.
        pill_labels: List of labels of the pills.
    """

    bg_img = bg_img.copy()
    h_bg, w_bg = bg_img.shape[0], bg_img.shape[1]
    comp_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)

    pill_areas: List[int] = []
    pill_labels: List[int] = []
    num_pills = np.random.randint(min_pills, max_pills + 1)
    idx = np.random.randint(len(pill_mask_paths))
    pill_img, mask = get_img_and_mask(pill_mask_paths[idx])

    for i in range(1, num_pills + 1):
        success: bool = False
        for _ in range(max_attempts):
            # Randomly sample a position for the pill.
            # The position is sampled from a normal distribution with mean at the center of the background image
            # and standard deviation of a quarter of the background image's width and height.
            x, y = np.random.normal(
                loc=(w_bg / 2, h_bg / 2), scale=(w_bg / 8, h_bg / 8), size=(2,)
            )
            x, y = np.clip(x, 0, w_bg), np.clip(y, 0, h_bg)

            # Resize and transform the pill image and mask.
            pill_img, mask = resize_and_transform_pill(pill_img, mask, 100, 100)

            # Add the pill to the background image.
            bg_img_prev, comp_mask_prev = bg_img.copy(), comp_mask.copy()
            bg_img, comp_mask, added_mask = add_pill_on_bg(
                bg_img, comp_mask, pill_img, mask, int(x), int(y), i
            )

            # Verify that the pill does not overlap with other pills too much.
            if verify_overlap(comp_mask, pill_areas, max_overlap):
                pill_areas.append(np.count_nonzero(added_mask))
                pill_labels.append(1)
                success = True
                break
            else:
                bg_img, comp_mask = bg_img_prev.copy(), comp_mask_prev.copy()

        if not success:
            break

    return bg_img, comp_mask, pill_labels, pill_areas
