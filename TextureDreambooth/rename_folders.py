# Given a directory, rename all folders in the directory that match with pattern "xxx_sks_20241022" to "xxx_sks".
import os, sys
import glob
import re
data_dir = "output"
pattern = "_sks_20241022"
for folder in glob.glob(os.path.join(data_dir, "*")):
    if pattern in folder:
        new_folder = re.sub(pattern, "_sks", folder)
        os.rename(folder, new_folder)
        print(f"Renamed {folder} to {new_folder}")