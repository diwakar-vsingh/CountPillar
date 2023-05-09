from pathlib import Path
from typing import Optional

import click
import cv2
from tqdm import tqdm

from countpillar.composition import create_pill_comp
from countpillar.io_utils import create_yolo_annotations, load_pill_mask_paths
from countpillar.object_overlay import generate_random_bg
from countpillar.transform import resize_bg


@click.command()
@click.option(
    "-p",
    "--pill-mask-path",
    default=Path("./data/pills/"),
    type=click.Path(exists=True),
    show_default=True,
    help="Path to the folder with pill masks",
)
@click.option(
    "-b",
    "--bg-img-path",
    default=None,
    type=click.Path(exists=True),
    show_default=True,
    help="Path to the background image",
)
@click.option(
    "-o",
    "--output-folder",
    default=Path("./dataset/synthetic/"),
    type=click.Path(),
    show_default=True,
    help="Path to the output folder",
)
@click.option(
    "-n",
    "--n-images",
    default=100,
    show_default=True,
    help="Number of images to generate",
)
@click.option(
    "-mp",
    "--min-pills",
    default=5,
    show_default=True,
    help="Minimum number of pills per image",
)
@click.option(
    "-MP",
    "--max-pills",
    default=50,
    show_default=True,
    help="Maximum number of pills per image",
)
@click.option(
    "-mo",
    "--max-overlap",
    default=0.2,
    show_default=True,
    help="Maximum overlap between pills",
)
@click.option(
    "-ma",
    "--max-attempts",
    default=10,
    show_default=True,
    help="Maximum number of attempts to place a pill",
)
@click.option(
    "-mb",
    "--min-bg-dim",
    default=1080,
    show_default=True,
    help="Minimum dimension of the background image",
)
@click.option(
    "-MB",
    "--max-bg-dim",
    default=1920,
    show_default=True,
    help="Maximum dimension of the background image",
)
def main(
    pill_mask_path: Path,
    bg_img_path: Optional[Path],
    output_folder: Path,
    n_images: int,
    min_pills: int,
    max_pills: int,
    max_overlap: float,
    max_attempts: int,
    min_bg_dim: int,
    max_bg_dim: int,
):
    # Load pill mask paths
    pill_mask_paths = load_pill_mask_paths(pill_mask_path)
    print(f"Found {len(pill_mask_paths)} pill masks.")

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "images").mkdir(parents=True, exist_ok=True)
    (output_folder / "labels").mkdir(parents=True, exist_ok=True)

    # Load and resize background image
    if bg_img_path is not None:
        bg_img = cv2.imread(str(bg_img_path))
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_img = resize_bg(bg_img, max_bg_dim, min_bg_dim)
    else:
        bg_img = generate_random_bg(min_bg_dim, max_bg_dim)

    # Generate images and annotations and save them
    for j in tqdm(range(n_images), desc="Generating images"):
        img_comp, mask_comp, labels_comp, _ = create_pill_comp(
            bg_img, pill_mask_paths, min_pills, max_pills, max_overlap, max_attempts
        )
        img_comp = cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR)

        anno_yolo = create_yolo_annotations(mask_comp, labels_comp)
        label_folder = output_folder / "labels"
        with (label_folder / f"{j}.txt").open("w") as f:
            for idx in range(len(anno_yolo)):
                f.write(" ".join(str(el) for el in anno_yolo[idx]) + "\n")
        cv2.imwrite(
            str(output_folder / "images" / f"{j}_{len(anno_yolo)}.jpg"), img_comp
        )

    print("Annotations are saved to the folder: ", label_folder)
    print("Images are saved to the folder: ", output_folder / "images")


if __name__ == "__main__":
    main()
