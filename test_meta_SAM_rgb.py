import os
import numpy as np
import cv2
import supervision as sv
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

home = os.path.expanduser("~")
IMAGE_PATH = os.path.join(home, "shared/testbed19")
CHECKPOINT_PATH = os.path.join(IMAGE_PATH, "checkpoint")


def save_png(image_bgr, result, outfile):
    # Annotate original image
    mask_annotator = sv.MaskAnnotator()
    print("Result: ", result)
    detections = sv.Detections.from_sam(result)
    image = mask_annotator.annotate(image_bgr, detections)

    # Convert to bytes
    image = image.astype(np.uint8)

    # Write
    cv2.imwrite(outfile, image)


def run_meta_sam(image_rgb):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE]()  # checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    # Generate mask
    mask_generator = SamAutomaticMaskGenerator(sam)
    result = mask_generator.generate(image_rgb)

    return result


def main():
    # Check input image exists
    infile = os.path.join(IMAGE_PATH, 'input_rgb.png')
    if not os.path.exists(infile):
        print("Failed to access {}".format(infile))
        return None

    # Read input file
    image_bgr = cv2.imread(infile)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Run the SAM model
    result = run_meta_sam(image_rgb)

    # Save to png file
    if len(result) > 0:
        outfile = os.path.join(IMAGE_PATH, 'output_rgb.png')
        if os.path.exists(outfile):
            os.remove(outfile)
        save_png(image_bgr, result, outfile)
    else:
        print("No result returned: {}".format(result))


if __name__ == "__main__":
    exit(main())
