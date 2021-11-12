'''
Sequentially name all images in a folder.
'''

import os, sys, glob
from shutil import copyfile

indir = sys.argv[1]
outdir = sys.argv[2]
if not os.path.exists(outdir):
    os.makedirs(outdir)

ext = ['png', 'jpg', 'jpeg']
files = []
[files.extend(glob.glob(indir + '/*.' + e)) for e in ext]
print("{} files found.".format(len(files)), files)

imgno=0
for file in files:
	copyfile(file, outdir+"/{}.jpg".format(imgno))
	imgno+=1

print("{} files copied.".format(imgno))