'''
Create gif from image files
'''

import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys, os, subprocess




indir = sys.argv[1]
outdir = sys.argv[2]


def create_blurred_images(img):
    if not os.path.exists('temp'):
        os.makedirs('temp')

    sigma=101
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.circle(mask, center=(290, 250), radius=100,
                  color=(255, 255, 255), thickness=-1)

    while sigma>0:
        blurred_img = cv2.GaussianBlur(img, (sigma, sigma), 0)
        out = np.where(mask==np.array([255, 255, 255]), img, blurred_img)
        cv2.imwrite("temp/{}.jpg".format(sigma), out)
        sigma-=2


#for each image in the input directory
for file in glob.glob(indir+"/*.jpg"):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    inimg = cv2.imread(file)
    blurred_imgs = create_blurred_images(inimg) # create the blurred image

    filenames= ['temp/{}.jpg'.format(i) for i in range(101,0,-2)]
    out_gif= '{}/{}-anim.gif'.format(outdir, os.path.basename(file))

    frames_in_anim = len(filenames)
    animation_loop_seconds = 2 #time in seconds for animation to loop one cycle
    seconds_per_frame = animation_loop_seconds / frames_in_anim
    frame_delay = str(int(seconds_per_frame * 100))

    command_list = ['convert', '-delay', frame_delay, '-loop', '1'] \
                     + filenames + [out_gif]
    subprocess.call(command_list)