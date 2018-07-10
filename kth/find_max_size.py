from kth.GraphFileOperations import *
from kth.FloorPlanGraph import *
import numpy as np

smallpath = "/home/bird/data/kth_floorplan_clean_categories/small"
middlepath = "/home/bird/data/kth_floorplan_clean_categories/middle"
largepath = "/home/bird/data/kth_floorplan_clean_categories/large"

if __name__ == '__main__':
    import sys
    import shutil

    if len(sys.argv) < 2:
        sys.exit('Usage: python {} <floorplan_dir>'.format(sys.argv[0]))

    floorplan_dir = sys.argv[1]
    graphs, graph_properties = GraphFileOperations.loadAllGraphsInFolder(sdir=floorplan_dir, rootNodeName="floor")

    small_num = 0
    middle_num = 0
    large_num = 0

    max_size = [0, 0]
    for g in graphs:
        size = [0, 0]
        size[0] = g.graph["attr"].maxx - g.graph["attr"].minx
        size[1] = g.graph["attr"].maxy - g.graph["attr"].miny
        resolution = g.graph["attr"].real_distance / g.graph["attr"].pixel_distance
        # size[0] = fp.graph.graph["attr"].maxx - fp.graph.graph["attr"].minx
        # size[1] = fp.graph.graph["attr"].maxy - fp.graph.graph["attr"].miny
        # resolution = fp.graph.graph["attr"].real_distance / fp.graph.graph["attr"].pixel_distance
        size_real = np.asarray(size) * resolution

        fullpath = g.graph["attr"].filepath
        head, filename = os.path.split(fullpath)
        head, parent_dir = os.path.split(head)

        if size_real[0] <= 35 and size_real[1] <= 35:
            # destdir = os.path.join(smallpath, parent_dir)
            destdir = smallpath
            small_num += 1
        elif size_real[0] <=70 and size_real[1] <= 70:
            # destdir = os.path.join(middlepath, parent_dir)
            destdir = middlepath
            middle_num += 1
        else:
            # destdir = os.path.join(largepath, parent_dir)
            destdir = largepath
            large_num += 1

        if not os.path.exists(destdir):
            os.mkdir(destdir)

        destfile = os.path.join(destdir, filename)
        if os.path.exists(destfile):
            print("confict file:")
            print(fullpath)
        shutil.copy2(fullpath, destdir)


        # generate bitmap
        basename, ext = os.path.splitext(filename)
        bitmap_name = os.path.join(destdir, basename+".bmp")
        floorplan = FloorPlanGraph()
        floorplan.graph = g
        img = floorplan.to_image(0.1, (1360, 1020))
        image = np.uint8(img * 255)
        cv2.imwrite(bitmap_name, image)

        # print("size of floorplan " + fp.graph["attr"].filepath + ":")
        print(size_real)

        if size_real[0] > max_size[0]:
            max_size[0] = size_real[0]
        if size_real[1] > max_size[1]:
            max_size[1] = size_real[1]

    print("max size:")
    print(max_size)

    print("small size:" + str(small_num))
    print("small size:" + str(middle_num))
    print("small size:" + str(large_num))
