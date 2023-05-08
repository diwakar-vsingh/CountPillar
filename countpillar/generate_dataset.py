from pathlib import Path

import click
import cv2
from tqdm import tqdm

from countpillar.composition import create_pill_comp
from countpillar.io_utils import create_yolo_annotations, load_pill_mask_paths


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
    "--bg_img-path",
    default=Path("./data/bg/plate1.jpg"),
    type=click.Path(exists=True),
    show_default=True,
    help="Path to the background image",
)
@click.option(
    "-o",
    "--output_folder",
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
    default=40,
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
def main(
    pill_mask_path: Path,
    bg_img_path: Path,
    output_folder: Path,
    n_images: int,
    min_pills: int,
    max_pills: int,
    max_overlap: float,
    max_attempts: int,
):
    # Load pill mask paths
    pill_mask_paths = load_pill_mask_paths(pill_mask_path)
    print(f"Found {len(pill_mask_paths)} pill masks.")

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "images").mkdir(parents=True, exist_ok=True)
    (output_folder / "labels").mkdir(parents=True, exist_ok=True)

    # Load background image
    bg_img = cv2.imread(str(bg_img_path))
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

    for j in tqdm(range(n_images), desc="Generating images"):
        img_comp, mask_comp, labels_comp, _ = create_pill_comp(
            bg_img, pill_mask_paths, min_pills, max_pills, max_overlap, max_attempts
        )
        img_comp = cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_folder / "images" / f"{j}.jpg"), img_comp)

        annotations_yolo = create_yolo_annotations(mask_comp, labels_comp)
        for idx in range(len(annotations_yolo)):
            label_folder = output_folder / "labels"
            with (label_folder / f"{j}.txt").open("a") as f:
                f.write(" ".join(str(el) for el in annotations_yolo[idx]) + "\n")

    print("Annotations are saved to the folder: ", output_folder / "labels")
    print("Images are saved to the folder: ", output_folder / "images")


if __name__ == "__main__":
    main()
