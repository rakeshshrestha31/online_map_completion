import sys

def fix_python_path():
    if sys.version_info.major == 3:
        sys.path = [i for i in sys.path if 'python2' not in i]
    elif sys.version_info.major == 3:
        sys.path = [i for i in sys.path if 'python3' not in i]

def add_current_path():
    print('adding path: ', __file__)
    sys.path.append(__file__)

