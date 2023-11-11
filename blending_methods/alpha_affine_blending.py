#import all required libraries

import os
import sys
import natsort
import glob

from scipy import ndimage
import skimage.io
from skimage import color

import numpy as np
import cv2 #as cv
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

import time


#class creation
class alphaaffine(object):

    #class constructor
    def __init__(self, srcimages, dstimages):
        self.srcimages = srcimages
        self.dstimages = dstimages


    #method to resize source image

    def resizesrc(self):

        self.src_resized_array = []

        for i in range(len(self.srcimages)):
            self.im_src_resized = cv2.resize(self.srcimages[i], (int(self.dstimages[1].shape[1]), int(self.dstimages[1].shape[0])))

            self.src_resized_array.append(self.im_src_resized)

        
        
    #method for four corner
    def fourcorner(self):

        self.im_out_array = []

        df = pd.read_csv(r'E:/python_programming/phd/billboard_seamlessintegration/csv_data/data4corner_set4.csv')

        #imagename

        imagename = df['image_name']

        #lt
        x1 = df['left_top_x1']
        y1 = df['left_top_y1']
        
        #rt
        x2 = df['right_top_x2']
        y2 = df['right_top_y2']

        
        #lb
        x3 = df['left_bottom_x3']
        y3 = df['left_bottom_y3']

        
        #rb
        x4 = df['right_bottom_x4']
        y4 = df['right_bottom_y4']


        #lt, lb, rt, rb
        #np.array([[0, 0], [0, 128], [128, 0], [128, 128]]) #tesco advert #np.array([[0, 0], [0, 512], [512, 0], [512, 512]]) #tesco, denos, stone
        #np.array([[8,31],[8,81], [123,31],[123,81]]) #scene_12 #np.array([[34, 125], [34, 323], [491, 125], [491, 323]]) #scene_12 512 x 512 image size
        #np.array([[40,24],[40,48], [90,24],[90,48]]) #scene_52 #np.array([[159, 95],[159, 191],[362, 95],[362, 191]]) #scene_52 #512 x 512 image size
        #np.array([[96, 32], [96, 146], [352, 32],  [352, 146]]) # scene_256 #512 x 512 image size
        


        
        for j in range(len(self.srcimages)):
         for i in range(len(self.dstimages)):
            #read four corner data from csv file
            #Four corners of the advert in source image
            pts_src = np.array([[96, 32],[96, 146],[352, 32],[352, 146]])
            
            #print('src image coordinates:', pts_src)

            # Four corners of the billboard in destination image.
            pts_dst = np.array([[x1[i], y1[i]], [x3[i], y3[i]] ,[x2[i], y2[i]], [x4[i], y4[i]]])

            pts_imagename = imagename[i]

            #print(pts_dst)
            #left-top, left-bottom,  right-top, right-bottom
            offset = (0, 66)

            #Calculate Homography
            h, status = cv2.findHomography(pts_src, pts_dst)
            #print(h)
            #print(status)

            # Warp source image to destination based on homography
            self.im_out = cv2.warpPerspective(self.src_resized_array[j], h, (self.dstimages[i].shape[1], self.dstimages[i].shape[0]))
            
        
            #print('Source image size:', self.im_out.shape[:-1])
            self.src_file_name = read_src_files[j]
            self.src_image_name = os.path.splitext(os.path.basename(self.src_file_name))[0]
            
            imout_imagename = self.src_image_name+ '_' + pts_imagename
            print('imout imagename:', imout_imagename)
            
            plt.axis('off')
            #plt.title('im_out')
            plt.imshow(self.im_out[:,:,::-1])
            #plt.savefig('E:/python_programming/phd/billboard_seamlessintegration/alpha_affine_blending/blend_output/affine_output/' +imout_imagename+ '', bbox_inches = 'tight', pad_inches = 0)
            #plt.show()
            
            self.im_out_array.append(self.im_out)


    #method to convert rbgba
    def convertrgba(self):

        
        self.foreground_bgr_array = []

        for i in range(len(self.im_out_array)):
            
            #denis, stone, tesco
            self.foreground_imout_rgba = cv2.cvtColor(self.im_out_array[i], cv2.COLOR_BGR2RGBA)
            #print('foreground_imout_rgba:', self.foreground_imout_rgba.shape)

            self.foreground_bgr = self.foreground_imout_rgba[:,:,0:3]
            self.foreground_alpha = self.foreground_imout_rgba[:,:,3]

            
            '''
            #bundle(scene_12), kid(scene_52)
            self.foreground_imsrc_rgba = cv2.cvtColor(self.src_resized_array[i], cv2.COLOR_BGR2RGBA)

            self.foreground_bgr = self.foreground_imsrc_rgba[:,:,0:3]
            self.foreground_alpha = self.foreground_imsrc_rgba[:,:,3]
            '''

            ##Define foreground image center and scale
            #self.foreground_center = (self.src_resized_array[i].shape[1] /2, self.src_resized_array[i].shape[0] /2)
            #self.foreground_scale = 1.0

            #create alpha mask for foreground image

            self.alpha_mask =  np.zeros_like(self.foreground_alpha)
            self.alpha_mask[self.foreground_alpha >0] = 1

            
            self.foreground_bgr_array.append(self.foreground_bgr)
            
        
    #method  for alpha_affine_blending
    def alphaaffineblending(self):

        self.blended_array = []


        
        for i in range(len(self.dstimages)):
              dst_file_name = read_dst_files[i]
              dst_image_name = os.path.splitext(os.path.basename(dst_file_name))[0]
              #print('src_image_name:', src_image_name)
              #print('dst_image_name:', dst_image_name)

              dst_src_file_name = dst_image_name+'_' + self.src_image_name 
              print('filename:',dst_src_file_name)
                                            
             #blend foreground and background using alpha mask
              self.blended = cv2.addWeighted(self.dstimages[i], 0.5, self.foreground_bgr_array[i], 1, 1)
              self.blended_rgb = cv2.cvtColor(self.blended, cv2.COLOR_RGBA2RGB)

              self.blended_array.append(self.blended)

              #save the blended image 
              #cv2.imwrite('./blend_output/alpha_affine/{dst_src_file_name}', self.blended)
              #cv2.imshow('blended', self.blended)

              plt.imshow(self.blended[:,:,::-1])
              plt.axis('off')
              #plt.savefig('E:/python_programming/phd/billboard_seamlessintegration/alpha_affine_blending/blend_output/alpha_affine_f/' +dst_src_file_name+ '' , bbox_inches = 'tight', pad_inches = 0)
              #plt.show() 


#Main method

if __name__ == "__main__":

    #set1: bundle, set2: kid, set3: denis, set4: stone, set5: tesco, set7: fuel
    start_time = time.time()
    
    src_image_path ='E:/python_programming/phd/billboard_seamlessintegration/images/foreground/set7/'
    dst_image_path ='E:/python_programming/phd/billboard_seamlessintegration/images/testdataset_background/testdataset4/'


    num_of_src_images = len(os.listdir(src_image_path))
    num_of_dst_images = len(os.listdir(dst_image_path))

    print('num of src images:', num_of_src_images)
    print('num of dst images:', num_of_dst_images)

    read_src_files = natsort.natsorted(glob.glob(src_image_path + "*.jpg"))
    read_dst_files = natsort.natsorted(glob.glob(dst_image_path + "*.jpg"))

    

    src_images = [skimage.io.imread(file)
                   for file in read_src_files]

    dst_images = [skimage.io.imread(file)
                   for file in read_dst_files]

   

    val = alphaaffine(src_images, dst_images)

    val.resizesrc()

    val.fourcorner()

    val.convertrgba()

    val.alphaaffineblending()    
          
    end_time = time.time()

    print(f"Time: {end_time - start_time} seconds")
