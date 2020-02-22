import numpy as np
import pandas as pd

import os
from fnmatch import fnmatch
from shutil import copyfile

# Parameters
root = '../Dataset_aug/'
root_out = '../Dataset/'

pattern = "*.png"

train_sub = "train"
val_sub = "val"
test1_sub = "test1"

test1_split = 0.90
val_split = 0.80


def grab_all_files(root = ".", pattern = "*"):
    paths = []
    filenames = []
    
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                paths.append(path)
                filenames.append(name)

    return paths, filenames


# REF: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

# Grab list of all files
im_paths, im_filenames = grab_all_files(root = root, pattern = pattern)
df_im = pd.DataFrame({'path': im_paths, 'filename':im_filenames})

# Add target folder based on random number
df_im['rand'] = np.random.random(df_im.shape[0])

def target_folder(r):
    if r>test1_split:
        f = test1_sub
    elif r>val_split:
        f = val_sub
    else:
        f = train_sub
    
    return f

df_im['target'] = df_im.apply(lambda row: target_folder(row['rand']), axis=1)

# Create target path and file name
df_im['class'] = df_im.apply(lambda row: splitall(row['path'])[2], axis=1)
df_im['subf1'] = df_im.apply(lambda row: splitall(row['path'])[3], axis=1)

def new_file_name(r):
    return r['class'] + '_' + r['subf1'] + '_' + r['filename']

def new_target_path(r):
    return os.path.join(root_out, r['target'], r['class'])

df_im['new_path'] = df_im.apply(lambda row: new_target_path(row), axis=1)
df_im['new_filename'] = df_im.apply(lambda row: new_file_name(row), axis=1)

# create Excel file of counts to verify split on classes
df_im.groupby('target')['class'].value_counts().to_excel('train_val_test1_split.xlsx')

# Final step - copy all files
for index, row in df_im.iterrows():
    if index % 10000 == 0:
        print(".", end = "")
        
    # Create folder if needed
    if not os.path.exists(row['new_path']):
        os.makedirs(row['new_path'])
        
    # Copy file
    src = os.path.join(row['path'], row['filename'])
    dst = os.path.join(row['new_path'], row['new_filename'])
    
    copyfile(src, dst)

print("")
print("Done!")



