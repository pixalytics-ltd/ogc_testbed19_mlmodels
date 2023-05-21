
import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

home = os.path.expanduser("~")
IMAGE_PATH = os.path.join(home, "shared/testbed19")
CHECKPOINT_PATH = os.path.join(IMAGE_PATH,"checkpoint")

def save_png(image, outfile):

    # Convert to bytes
    image = image.astype(np.uint8)
    # Write
    cv2.imwrite(outfile,image)


def run_SAM():

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE]()#checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    # Input and output file
    infile = os.path.join(IMAGE_PATH,'input_rgb.png')
    if not os.path.exists(infile):
        print("Failed to access {}".format(infile))
        return None

    # Read input file
    image_bgr = cv2.imread(infile)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Generate mask
    mask_generator = SamAutomaticMaskGenerator(sam)
    result = mask_generator.generate(image_rgb)

    return result



def main():

    # Run the SAM model
    result = run_SAM()

    # Save to png file
    if result is not None:
        outfile = os.path.join(IMAGE_PATH,'output_rgb.png')
        if os.path.exists(outfile):
            os.remove(outfile)
        save_png(result, outfile)



if __name__ == "__main__":
    exit(main())
