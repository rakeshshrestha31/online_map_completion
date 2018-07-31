
import glob
import os
import re
import cv2


def removeBlur(src_dir):
    reconstruction_files = glob.glob(os.path.join(src_dir, 'reconstruction_*.png'), recursive=True)

    regex = re.compile(r'.*reconstruction_(\d+).png')
    file_indices = [
        re.search(regex, reconstruction_file).group(1)
        for reconstruction_file in reconstruction_files
    ]

    new_dir = os.path.join(src_dir, "reconstruction_deblur")

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    deBlur_files = [
        os.path.join(new_dir, 'reconstruction_deblur_' + str(file_indices[i]) + '.png')
        for i in range(len(file_indices))
    ]

    for i in range(len(reconstruction_files)):
        file = reconstruction_files[i]
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)  # keep 4-channels of image

        # threshold blue channel (obstacle channel)
        ret, deblur_channel = cv2.threshold(image[:, :, 0], 128, 255, cv2.THRESH_BINARY)
        image[:, :, 0] = deblur_channel

        cv2.imwrite(deBlur_files[i], image)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Remove blur obstacles in predicted area')
    parser.add_argument('--src', dest='input_directory', metavar="directory",
                        type=str, help="Source directory / the results directory")
    args = parser.parse_args()
    removeBlur(args.input_directory)