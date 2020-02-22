import numpy as np
import pandas as pd

import PIL
from PIL import Image
import os
from fnmatch import fnmatch

# Parameters
root = '../Dataset_base/'
pattern = "*.png"
fnappend = "_rot"
subpath = "rots"


def rotate_image(im, deg):
    return im.rotate(deg, resample=PIL.Image.BICUBIC)
    

def grab_all_files(root = ".", pattern = "*"):
    paths = []
    filenames = []
    
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                paths.append(path)
                
                # ensure file does not contain the key fnappend
                if (name.find(fnappend) == -1):
                    filenames.append(name)

    return paths, filenames


def new_filename(row):
    f = row['filename']
    rot = row['rot']
    
    fn, fext = os.path.splitext(f)
    fnew = fn + fnappend + str(rot) + fext
    
    return fnew


def new_path(row):
    path = row['path']
    
    return os.path.join(path, subpath)


def setup_work():
    
    #1. Grab list of all files
    im_paths, im_filenames = grab_all_files(root = root, pattern = pattern)
    df_im = pd.DataFrame({'path': im_paths, 'filename':im_filenames})
    
    #2. Create rotations needed 10 to 350 in steps of 10
    rot = np.arange(10, 360, 10).tolist()
    df_rot = pd.DataFrame({'rot': rot})
    
    #3. Create target dataset of rotations
    #     cross join df_im and df_rot
    df_im['key'] = 0
    df_rot['key'] = 0
    df_new = df_im.merge(df_rot, how='left', on = 'key')
    df_new.drop('key', 1, inplace=True)
    
    #4. Add new file names
    df_new['new_filename'] = df_new.apply(new_filename, axis=1)
    
    #5. Add sub directory
    df_new['new_path'] = df_new.apply(new_path, axis=1)
    
    return df_new


df_new = setup_work()

# Generate all images!
f_prev = ""

for index, row in df_new.iterrows():
    if index % 10000 == 0:
        print(".", end = "")
        
    f = os.path.join(row['path'], row['filename'])
    
    # Read image if new (ie not loaded in memory)
    if (f != f_prev):
        im = Image.open(f)
        f_prev = f
        
    # Rotate image
    im_rot = rotate_image(im, row['rot'])
    
    # Create sub dir if not exists
    if not os.path.exists(row['new_path']):
        os.makedirs(row['new_path'])
    
    # Save the image
    fs = os.path.join(row['new_path'], row['new_filename'])
    im_rot.save(fs)
    
print("")
print("Done!")

