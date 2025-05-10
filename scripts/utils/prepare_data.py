import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from config.params import *


def coco_to_masks(
    annotation_path: str,
    image_dir: str,
    output_dir: str,
    category_id: int = None,
) -> None:
    """
    Convert COCO annotations to binary segmentation masks.

    Args:
        annotation_path (str): Path to COCO annotation JSON file.
        image_dir (str): Directory containing input images.
        output_dir (str): Directory to save output masks.
        category_id (int, optional): Category ID to process. Defaults to first category.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize COCO API
    coco = COCO(annotation_path)
    img_ids = coco.getImgIds()

    # Select category ID if not provided
    if category_id is None:
        cats = coco.loadCats(coco.getCatIds())
        category_id = cats[0]["id"]
        print(f"Using category ID: {category_id} ({cats[0]['name']})")

    problematic_images = []

    # Process each image
    for img_id in tqdm(img_ids, desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        image_name = img_info["file_name"]
        print(f"\nProcessing image: {image_name}")

        # Load annotations for the image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=category_id)
        anns = coco.loadAnns(ann_ids)

        # Initialize binary mask
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        has_valid_annotations = False

        # Process each annotation
        for ann in anns:
            try:
                if ann["iscrowd"]:
                    # Handle RLE format for crowd regions
                    if isinstance(ann["segmentation"]["counts"], list):
                        rle = coco.annToRLE(ann)
                    else:
                        rle = ann["segmentation"]
                    m = coco.annToMask(
                        {
                            "size": [img_info["height"], img_info["width"]],
                            "counts": rle["counts"],
                        }
                    )
                else:
                    # Handle polygon or RLE format
                    if isinstance(ann["segmentation"], list):
                        m = coco.annToMask(ann)
                    else:
                        m = coco.annToMask(
                            {
                                "size": [img_info["height"], img_info["width"]],
                                "counts": ann["segmentation"]["counts"],
                            }
                        )

                mask = np.maximum(mask, m)
                has_valid_annotations = True
            except Exception as e:
                print(f"  Error processing annotation {ann['id']} in {image_name}: {str(e)}")
                problematic_images.append((image_name, ann["id"], str(e)))
                continue

        if not has_valid_annotations:
            print(f"  Warning: No valid annotations found for {image_name}")
            problematic_images.append((image_name, None, "No valid annotations"))
            continue

        # Save binary mask
        mask_filename = os.path.splitext(image_name)[0] + ".png"
        output_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(output_path, mask * 255)

    # Print summary of problematic images
    if problematic_images:
        print("\n\nSummary of problematic images/annotations:")
        for img_name, ann_id, error in set(problematic_images):
            if ann_id is not None:
                print(f"- Image: {img_name}, Annotation ID: {ann_id}, Error: {error}")
            else:
                print(f"- Image: {img_name}, Error: {error}")
    else:
        print("\nAll images processed successfully with no errors")


if __name__ == "__main__":
    # Process training dataset
    coco_to_masks(
        annotation_path=os.path.join(ANN_PATH, "instances_train2017.json"),
        image_dir=TRAIN_IMG_DIR,
        output_dir=TRAIN_MASK_DIR,
    )

    # Process validation dataset
    coco_to_masks(
        annotation_path=os.path.join(ANN_PATH, "instances_val2017.json"),
        image_dir=VAL_IMG_DIR,
        output_dir=VAL_MASK_DIR,
    )