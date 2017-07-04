##
import os
from shutil import copyfile
##

##
path_to_filenames = "/home/sam/data/baggage/full/ImageSets/baggage6/test.txt"

with open(path_to_filenames) as f:
    filenames = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
filenames = [x.strip() for x in filenames]

##
inp_dir = "/home/sam/data/baggage/full/Images"
tar_dir = "/home/sam/Desktop/t/journal"

for fname in filenames:
	copyfile(src=os.path.join(inp_dir, fname+'.jpg'),
	         dst=os.path.join(tar_dir, fname+'.jpg'))
