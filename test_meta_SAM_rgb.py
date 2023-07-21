import os
import sys
import numpy as np
import pickle
import cv2
import h5py
from sklearn import preprocessing
from sklearn.decomposition import PCA
import supervision as sv
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

home = os.path.expanduser("~")
ROOT = os.path.join(home, "shared/testbed19/")
IMAGE_PATH = os.path.join(ROOT, "CHRIS")
CHECKPOINT_PATH = os.path.join(ROOT, "sam_vit_h_4b8939.pth")

# global variables
n_components = 3


def read_chris(infile):
    numbands = 62
    first = True

    f = h5py.File(infile, 'r', libver='latest')

    # loop through all bands and read data
    for b in range(numbands):
        bnum = str(b + 1)
        if "TOA" in infile:
            bname = 'bands/toa_refl_' + bnum
        else:
            bname = 'bands/radiance_' + bnum

        try:
            dset = f[bname]
        except:
            print("{} does not exist, skipping".format(bname))
            continue

        # Setup band array
        if first:
            # Setup data array
            rows, cols = dset.shape
            data = np.zeros((numbands, rows, cols), dtype=float)
            print("Band data: ", data.shape)

            # Everything setup    
            first = False

        # Save data array
        data[b, :, :] = dset[:, :]
        del dset

    # Close file
    f.close()

    return data


def save_png(image_bgr, detections, outfile, annotate=True):
    if annotate:
        # Annotate original image with detections
        mask_annotator = sv.MaskAnnotator()
        image = mask_annotator.annotate(image_bgr, detections)
    else:
        image = image_bgr

    # Convert to bytes
    image = image.astype(np.uint8)

    # Write
    cv2.imwrite(outfile, image)

    return image


def run_meta_sam(image_rgb):
    # Check whether GPUs are accessible
    # os.system("nvidia-smi")

    # Setup where to use GPUs and model types
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    # Generate mask
    mask_generator = SamAutomaticMaskGenerator(sam)
    result = mask_generator.generate(image_rgb)

    return result


def main():
    # Whether to run for all bands using PCA
    pca = False
    hloop = True

    # Setup filenames
    if pca:
        outimg = 'output_hyp_pca.png'
    elif hloop:
        outimg = 'output_hyp_loops.png'
    else:
        outimg = 'output_rgb.png'
    outpng = os.path.join(IMAGE_PATH, outimg)
    restore = False
    outpi = outpng.replace(".png",".obj")

    if pca or hloop:
        inhyp = os.path.join(IMAGE_PATH, 'CHRIS_BR_220607_8FDD_41_NR_TOA_REFL_GC.h5')
        if not os.path.exists(inhyp):
            print("Failed to access {}".format(inhyp))
            return None

    # Check input image exists
    infile = os.path.join(IMAGE_PATH, 'input_rgb.png')
    if not os.path.exists(infile):
        print("Failed to access {}".format(infile))
        return None

    # Read RGB input file
    image_bgr = cv2.imread(infile)

    # Read TIFF input file
    if pca:
        if os.path.exists(outpi):
            restore = True
            pifile = open(outpi, 'rb')
            detections = pickle.load(pifile)

        else:
            # Read in CHRIS data from HDF file
            data = read_chris(inhyp)
            bands, ny, nx = data.shape

            # Flatten to 2D array suitable for PCA
            # features (bands) as columns and (pixels) samples
            data_2d = data.reshape(bands, ny * nx)
            print("data: {} flattened data: {}".format(data.shape, data_2d.shape))

            # Normalise the data
            scaler = preprocessing.MinMaxScaler()
            scaled = scaler.fit_transform(data_2d)

            # Run PCA
            pca = PCA(n_components=n_components)
            result = pca.fit_transform(scaled)

            # Reconstruct original images
            # pca_data_d2 = pca.inverse_transform(result)
            pca_comp = pca.components_
            minv = np.nanmin(pca_comp)
            maxv = np.nanmax(pca_comp)
            print("PCA output: {} min {:.3f} max {:.3f}".format(pca_comp.shape, minv, maxv))

            # Reconstruct as 2D image
            pca_data_d2 = pca_comp.reshape(n_components, ny, nx)

            # Reorder then apply scaling so can be integers
            image_pca = np.moveaxis(pca_data_d2, 0, -1)
            image_pca = np.subtract(image_pca, minv)
            image_pca = np.multiply(image_pca, 255 / (maxv - minv))

            print("PCA input for SAM model: {} min {} max {}".format(image_pca.shape, np.nanmin(image_pca),
                                                                     np.nanmax(image_pca)))

            # Save PCA components
            pca_sav = save_png(image_pca, image_pca, os.path.join(IMAGE_PATH, 'output_pca.png'), annotate=False)
            # sys.exit(1)

            # Run the SAM model
            print("Running SAM model")
            result = run_meta_sam(image_pca.astype(np.uint8))
            # print("Result: ", result)

            # Extract detections
            detections = sv.Detections.from_sam(result)


    elif hloop:  # Loop multiple band combinations

        if os.path.exists(outpi):
            restore = True
            pifile = open(outpi, 'rb')
            detections = pickle.load(pifile)

        else:

            # Read in CHRIS data from HDF file
            data = read_chris(inhyp)
            bands, ny, nx = data.shape

            # Reorder then apply scaling so can be integers
            minv = np.nanmin(data)
            maxv = np.nanmax(data)
            hyp_data = np.moveaxis(data, 0, -1)
            hyp_data = np.subtract(hyp_data, minv)
            hyp_data = np.multiply(hyp_data, 255 / (maxv - minv))
            del data

            loop = 0
            topband = 0
            while topband < bands:

                # Extract 3 bands
                # BGR is 2, 13, 23 so seperate 3 bands similary
                i = [[loop], [loop + 10 + 1], [loop + 20 + 1]]
                topband = loop + 20 + 1
                image_rgb = hyp_data[:, :, i].reshape(ny, nx, 3)

                # Run the SAM model
                print("Running SAM model for iteration {}".format(loop))
                result = run_meta_sam(image_rgb.astype(np.uint8))
                # print("Result: ", result)

                # Extract detections
                iter_detections = sv.Detections.from_sam(result)

                # Merge detections
                if loop == 0:
                    detections = iter_detections
                else:
                    detections = sv.Detections.merge([detections, iter_detections])

                # Iterate loop
                loop+=1

    else:
        if os.path.exists(outpi):
            restore = True
            pifile = open(outpi, 'rb')
            detections = pickle.load(pifile)

        else:

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            print("RGB input for SAM model: {} min {} max {}".format(image_rgb.shape, np.nanmin(image_rgb), np.nanmax(image_rgb)))

            # Run the SAM model
            print("Running SAM model")
            result = run_meta_sam(image_rgb)
            # print("Result: ", result)

            # Extract detections
            detections = sv.Detections.from_sam(result)

    # Save to png file
    if len(detections) > 0:
        if os.path.exists(outpng):
            os.remove(outpng)

        # Save annotated png
        annotated_image = save_png(image_bgr, detections, outpng)

        # Display results
        sv.plot_images_grid(
            images=[image_bgr, annotated_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image'])

        # Save the detections
        if not restore:
            pifile = open(outpi, 'wb')
            pickle.dump(detections, pifile)

    else:
        print("No result returned: {}".format(result))


if __name__ == "__main__":
    exit(main())
