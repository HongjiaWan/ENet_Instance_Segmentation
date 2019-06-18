#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode the ground truth classes and the
# individual instance of that classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
# 
# With this tool, you can generate option
#   d) *instanceTrainIds.png  : the class and the instance are encoded by an instance training ID
# This encoding might come handy for training purposes. You can use
# the file labes.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Please refer to 'json2instanceImg.py' for an explanation of instance IDs.
#
# Uses the converter tool in 'json2instanceImg.py'
# Uses the mapping defined in 'labels.py'
#

# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys

from PIL import Image
import numpy as np
import torchvision.transforms as transforms

import torch

# cityscapes imports

from json2instanceImg import json2instanceImg


# The original ain method
'''
def main():
    # Where to look for Cityscapes
    
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    
    # how to search for all ground truth
    cityscapesPath = '/home/wan/PyTorch_ENet/label/train'
    searchFine   = os.path.join( cityscapesPath, "*" , "*_gt*_polygons.json" )
    #searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
  

    # concatenate fine and coarse
    files = filesFine 
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything


    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # create the output filename
        dst = f.replace( "_polygons.json" , "_instanceIdsImg.png" )

        # do the conversion
        try:
            json2instanceImg( f , dst , "ids" )
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()
'''
# the self-defined main method
def main():
    # Where to look for Cityscapes
    '''
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    '''
    # how to search for all ground truth
    cityscapesPath = '/home/wan/PyTorch_ENet/label/val'
    searchFine   = os.path.join( cityscapesPath, "*" , "*_gt*_instanceIds.png" )

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
  

    # concatenate fine and coarse
    files = filesFine 
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    label_transform = transforms.Compose([
        transforms.Resize((512, 1024))
    ])

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        dst = f.replace( "_instanceIds.png" , "_ResizedInstIds.png" )
        try:
            label = Image.open(f)
            label = label_transform(label)
            label.save(dst)
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()
# call the main
if __name__ == "__main__":
    main()
