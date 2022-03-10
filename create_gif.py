'''
Create gif from image files
'''

import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys, os, subprocess
from pathlib import Path
import csv


indir = sys.argv[1]
outdir = sys.argv[2]
box_dict_path = sys.argv[3]

def create_masked_images(img, face_center, radius, outfile):
    mask = np.full(img.shape, 1, dtype=np.uint8)
    mask = cv2.circle(mask, center=face_center, radius=radius,
                      color=(255, 255, 255), thickness=-1)
    out = np.where(mask==np.array([255, 255, 255]), img, mask)
    cv2.imwrite(outfile, out)
    

def create_black_images(img, face_center, radius, frame_count=101):
    mask = np.full(img.shape, 1, dtype=np.uint8)
    mask = cv2.circle(mask, center=face_center, radius=radius,
                      color=(255, 255, 255), thickness=-1)
    temp = np.where(mask==np.array([255, 255, 255]), img, mask)

    for i in range(frame_count):
        out = cv2.addWeighted(img, i/(frame_count-1), temp, 1-i/(frame_count-1), 0)

        out = np.where(mask==np.array([255, 255, 255]), img, out)
        cv2.imwrite("temp/{}.jpg".format(i), out)
        i+=1

def create_blurred_images(img):

    sigma=101
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.circle(mask, center=(290, 250), radius=100,
                  color=(255, 255, 255), thickness=-1)

    while sigma>0:
        blurred_img = cv2.GaussianBlur(img, (sigma, sigma), 0)
        out = np.where(mask==np.array([255, 255, 255]), img, blurred_img)
        cv2.imwrite("temp/{}.jpg".format(sigma), out)
        sigma-=2


box_dict={}
with open(box_dict_path) as csv_file:
    reader = csv.reader(csv_file)
    for box in list(reader):
        box_dict[box[0]]=[int(box[1]), int(box[2]), int(box[3]), int(box[4])]

if not os.path.exists(outdir):
    os.makedirs(outdir)
if not os.path.exists(outdir+'/masked-images/'):
    os.makedirs(outdir+'/masked-images/')
if not os.path.exists(outdir+'/gifs/'):
    os.makedirs(outdir+'/gifs/')
if not os.path.exists(outdir+'/temp'):
    os.makedirs(outdir+'/temp')

# for each image in the input directory
for file in glob.glob(indir+"/*.jpg"):
    
    print(file, box_dict[os.path.basename(file)])
    box =box_dict[os.path.basename(file)]
    x,y,w,h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    face_center = (int(x+w/2), int(y+h/2))
    radius = int(w/2 if w>h else h/2) +10

    inimg = cv2.imread(file)

    
    blurred_imgs = create_masked_images(inimg, face_center, radius, 
                outfile="{}/masked-images/{}".format(outdir,os.path.basename(file))) 
    # else: # create gifs from masked images
    frames_in_anim = 21
    blurred_imgs = create_black_images(inimg, face_center, radius, frames_in_anim) 
    filenames= ['temp/{}.jpg'.format(i) for i in range(frames_in_anim)]
    out_gif= '{}/gifs/{}-anim.gif'.format(outdir, Path(file).stem)

    animation_loop_seconds = 3 #time in seconds for animation to loop one cycle
    seconds_per_frame = animation_loop_seconds / frames_in_anim
    frame_delay = str(int(seconds_per_frame * 100))

    command_list = ['convert', '-delay', frame_delay, '-loop', '1'] \
                     + filenames + [out_gif]
    subprocess.call(command_list)