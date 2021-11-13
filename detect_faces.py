'''
Detect face in an image and draw a rectangle around the face.
Limitations
- only work for jpg files
'''

import glob
import cv2
import sys, os
from mtcnn import MTCNN
import csv

detector = MTCNN()

indir=sys.argv[1]
outdir=sys.argv[2]

ext = ['png', 'jpg']
# files = [(glob.glob( indir + '*.' + e)) for e in ext]

if not os.path.exists(outdir):
    os.makedirs(outdir)

box_list=[]
# for file in files:
for file in glob.glob(indir+"/*.jpg"):
    print(file)
    # Read the input image
    img = cv2.imread('{}'.format(file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = detector.detect_faces(img)[0]
    x,y,w,h=face['box']
    box_list.append([os.path.basename(file), x, y, w, h])
    # box_dict[os.path.basename(file)]=face['box']

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite(outdir+'/{}'.format(os.path.basename(file)), 
              cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

with open('boxes.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    writer.writerows(box_list)
    # for key, value in box_dict.items():
    #     writer.writerow([key, value])
