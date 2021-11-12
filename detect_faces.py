'''
Detect face in an image and draw a rectangle around the face.
Limitations
- only work for jpg files
'''

import glob
import cv2
import sys
from mtcnn import MTCNN
import os

detector = MTCNN()

indir=sys.argv[1]
outdir=sys.argv[2]

ext = ['png', 'jpg']
# files = [(glob.glob( indir + '*.' + e)) for e in ext]

if not os.path.exists(outdir):
    os.makedirs(outdir)

# for file in files:
for file in glob.glob(indir+"/*.jpg"):
    # Read the input image
    img = cv2.imread('{}'.format(file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = detector.detect_faces(img)[0]
    x,y,w,h=face['box']

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite(outdir+'/{}'.format(os.path.basename(file)), 
              cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


