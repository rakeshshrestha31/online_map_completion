import glob
import os
import json


def removeNullJson(basedir):
    info_files = glob.glob(os.path.join(basedir, '**', 'info*.json'), recursive=True)
    for info_file in info_files:
        f = open(info_file)
        info = json.load(f)
        f.close()
        if info["BoundingBoxes"] is None or info["Frontiers"] is None:
            os.remove(info_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Remove Invalid Json')

    parser.add_argument(
        'dataset_dir', type=str, default='.',
        metavar='S', help='dataset directory'
    )

    args = parser.parse_args()

    removeNullJson(args.dataset_dir)
