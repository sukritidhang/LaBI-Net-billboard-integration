#import all required libraries

import os
import sys
import natsort
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import ndimage
import skimage.io
from skimage import color

from PIL import Image

import time

#class creation

class cutpasteaffine(object):

    #class creation
    def __init__(self, imoutimages, dstimages, maskimages):
        self.imoutimages =  imoutimages
        self.dstimages = dstimages
        self.maskimages = maskimages

    
    #Method for direct cut paste affine blending
    def cutpasteaffineblending(self):

            #print(len(self.srcimages))
            self.blended_array = []

            imout_name = ['bundle.jpg', 'denis.jpg', 'kids.jpg', 'stone.jpg', 'tesco.jpg', 'ipad.jpg', 'fuel.jpg']
            
            for i in range(len(self.dstimages)):

                
                imout_image_name = imout_name[6]
                
                dst_file_name = read_dst_files[i]
                dst_image_name = os.path.splitext(os.path.basename(dst_file_name))[0]

                dst_src_filename = dst_image_name + '_' + imout_image_name
                print('filename:', dst_src_filename)
                

                self.dstimages[i].paste(self.imoutimages[i], self.maskimages[i])
                #self.blended_array.append(self.blendedimage)

                #cv2.imwrite(f'./blended_output/cutpaste_affine/{dst_src_filename}.jpg', self.dstimages[i])

                #self.dstimages[i].show()
                
                
                #saveblended figure
                plt.imshow(self.dstimages[i])
                plt.axis('off')
                #plt.savefig('E:/python_programming/phd/billboard_seamlessintegration/cutpaste_affine_blending/blend_output/cutpaste_affine_xx/' +dst_src_filename+ '', bbox_inches = 'tight', pad_inches = 0)
                #plt.show()
                
            

            
if  __name__ == "__main__":

    start_time = time.time()
    
    imout_image_path = 'E:/python_programming/phd/billboard_seamlessintegration/images/testdataset_imout/imoutdataset_fuel/'
    dst_image_path = 'E:/python_programming/phd/billboard_seamlessintegration/images/testdataset_background/resized_images_single_testdataset/'
    mask_image_path = 'E:/python_programming/phd/billboard_seamlessintegration/images/testdataset_mask/resized_masks_single_testdataset/'

    num_of_imout_images = len(os.listdir(imout_image_path))
    num_of_dst_images = len(os.listdir(dst_image_path))
    num_of_mask_images = len(os.listdir(mask_image_path))                                                        

    print('num of imout images:', num_of_imout_images)
    print('num of dst images:', num_of_dst_images)
    print('num of mask images:', num_of_mask_images)                                                        

    read_imout_files = natsort.natsorted(glob.glob(imout_image_path + "*.jpg"))
    read_dst_files = natsort.natsorted(glob.glob(dst_image_path + "*.jpg"))
    read_mask_files = natsort.natsorted(glob.glob(mask_image_path + "*.jpg"))
                                                        

    

    imout_images = [Image.open(file)
                   for file in read_imout_files]#skimage.io.imread

    dst_images = [Image.open(file)
                   for file in read_dst_files]

    mask_images = [Image.open(file).convert('L')
                    for file in read_mask_files]#convert to 8-bit grayscale image
    


    val =  cutpasteaffine(imout_images, dst_images, mask_images)

   
    val.cutpasteaffineblending()

    end_time = time.time()

    print(f"Time: {end_time - start_time} seconds")
    
