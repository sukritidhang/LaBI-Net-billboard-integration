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

import time

#class creation
class poissaffine(object):

    #class constructor
    def __init__(self, srcimages, dstimages, mskimages):
        self.srcimages = srcimages
        self.dstimages = dstimages
        self.mskimages = mskimages
            
    #method to resize
    def srcresize(self):

        #self.src_resized_array = []
        

        #for i in range(len(self.srcimages)):
        self.im_src_resized = cv2.resize(self.srcimages[0], (int(self.dstimages[1].shape[1]), int(self.dstimages[1].shape[0])))

        #self.src_resized_array.append(self.im_src_resized)


            
    #method for four corner
    def fourcorner(self):

        self.im_out_array = []
        self.pts_filename_array = []
        self.center_array = []

        df = pd.read_csv(r'E:/python_programming/phd/billboard_seamlessintegration/csv_data/data4corner_set3.csv')

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

        
        #for j in range(len(self.srcimages)):
        for i in range(len(self.dstimages)):
            #read four corner data from csv file
            #np.array([[0, 0], [0, 512], [512, 0], [512, 512]]) #tesco, denos, stone
            #np.array([[34, 125], [34, 323], [491, 125], [491, 323]]) #scene_12 512 x 512 image size
            #np.array([[159, 95],[159, 191],[362, 95],[362, 191]]) #scene_52 #512 x 512 image size
            #np.array([[96, 32], [96, 146], [352, 32],  [352, 146]]) # scene_256 #512 x 512 image size
        
            #Four corners of the advert in source image
            pts_src =  np.array([[34, 125], [34, 323], [491, 125], [491, 323]]) 
            #print('src image coordinates:', pts_src)

            # Four corners of the billboard in destination image.
            pts_dst = np.array([[x1[i], y1[i]], [x3[i], y3[i]] ,[x2[i], y2[i]], [x4[i], y4[i]]])

            self.X = x1[i] + x2[i]
            self.Y = y1[i] + y3[i]
            
            self.center = (self.X//2, self.Y//2)#(x1+x2//2, y1+y3//2)
            
            print('center:', self.center)

            pts_imagename = imagename[i]

            #print(pts_dst)
            #left-top, left-bottom,  right-top, right-bottom
            offset = (0, 66)

            #Calculate Homography
            h, status = cv2.findHomography(pts_src, pts_dst)
            #print(h)
            #print(status)

            # Warp source image to destination based on homography
            self.im_out = cv2.warpPerspective(self.im_src_resized, h, (self.dstimages[i].shape[1], self.dstimages[i].shape[0]))
            
        
            #print('Source image size:', self.im_out.shape[:-1])


            self.src_file_name = read_src_files[0]
            self.src_image_name = os.path.basename(self.src_file_name)[0]

            self.pts_filename = self.src_image_name + pts_imagename

            print('im out imagename:', self.pts_filename)
            
            plt.axis('off')
            #plt.title('im_out')
            plt.imshow(self.im_out[:,:,::-1])
            #plt.savefig('E:/python_programming/phd/billboard_seamlessintegration/poison_affine_blending/blend_output/affine_output/' +self.pts_filename+ '', bbox_inches = 'tight', pad_inches = 0)
            #plt.show()
            
            self.im_out_array.append(self.im_out)
            self.pts_filename_array.append(self.pts_filename)
            self.center_array.append(self.center)
            
            with open('E:/python_programming/phd/billboard_seamlessintegration/poison_affine_blending/center_value_denis.txt', 'a')as f:
                print("Image num:", self.pts_filename, file =f)
                print('center', self.center_array[i], file = f)
                

            
            
    
    def createmask(self):

        #create the mask of the source image for bundle and kids
        self.im_out_mask_array = []
        
        
        df = pd.read_csv(r'E:/python_programming/phd/billboard_seamlessintegration/csv_data/data4corner_set3.csv')

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
        
        
        for i in range(len(self.im_out_array)):

            #lt, rt, rb, lb
            #np.array([[0, 0], [512, 0], [512, 512], [0, 512]]) #tesco, denos, stone advert
            #np.array([[34, 125], [491, 125], [491, 323], [34, 323]]) #scene_12 512 x 512 image size
            #np.array([[159, 95],[362, 95],[362, 191],[159, 191]]) #scene_52 #512 x 512 image size

            # Create a rough mask around the im_out.
            self.im_out_mask = np.zeros(self.im_src_resized.shape, self.im_src_resized.dtype)#np.zeros(self.im_out_array[i].shape, self.im_out_array[i].dtype) 
            #lt,rt, rb, lb #mask creation based on destination four corners 
            points = np.array([[x1[i], y1[i]],[x2[i], y2[i]], [x4[i], y4[i]], [x3[i], y3[i]]])
            self.im_mask = cv2.fillPoly(self.im_out_mask, pts = [points], color = (255, 255, 255))
            #print('mask image shape:', self.im_mask.shape)
            #print('mask image shape:', self.im_mask.dtype)#uint8
            plt.imshow(self.im_out)
            
            
            self.pts_imagename = self.pts_filename_array[i]

            print('mask imagename:', self.pts_imagename)

            plt.axis('off')
            #plt.title('im_out')
            plt.imshow(self.im_mask)
            #plt.savefig('E:/python_programming/phd/billboard_seamlessintegration/poison_affine_blending/blend_output/src_mask_output/' +self.pts_imagename+ '', bbox_inches = 'tight', pad_inches = 0)
            

            self.im_out_mask_array.append(self.im_mask)
            
     
    #method  for poisson_affine_blending
    def poisaffineblending(self):

        self.blended_array = []

        
        '''
        print(len(self.dstimages))
        for i in range(len(self.dstimages)):
            center = (int(self.dstimages[i].shape[0]/2), int(self.dstimages[i].shape[1]/2))
            #center = (250, 230)
            #print(center)#location of the center of the source image in the destination image

            #(290, 180) #for scene_51_12, scene_51_tesco, scene_51_denis, scene_51_stone, scene_51_52
            #(250,230) #for scene_0_12, scene_0_tesco, scene_0_denis, scne_0_stone, scene_0_52  #(x, y) 
        '''
            
         
        
        #for i in range(len(self.dstimages)):
        for j in range(len(self.dstimages)):
              
                src_file_name = read_src_files[0]
                src_image_name = os.path.basename(src_file_name)
                
                dst_file_name = read_dst_files[j]
                dst_image_name = os.path.splitext(os.path.basename(dst_file_name))[0]
                #print('src_image_name:', src_image_name)
                #print('dst_image_name:', dst_image_name)

                self.dst_src_file_name = dst_image_name+'_' + src_image_name 
                print('filename:',self.dst_src_file_name)
                '''
                with open('E:/python_programming/phd/billboard_seamlessintegration/poison_affine_blending/center_value.txt', 'a')as f:
                    print("Image num:", self.dst_src_file_name, file =f)
                    print('center', self.center_array[j], file = f)
                '''
            
                #Seamlessly clone src into dst and put the results in output
                self.blended = cv2.seamlessClone(self.im_out_array[j], self.dstimages[j], self.im_out_mask_array[j], self.center_array[j], cv2.NORMAL_CLONE)
                

                self.blended_array.append(self.blended)

                plt.imshow(self.blended[:,:,::-1])
                plt.axis('off')
                #plt.savefig('E:/python_programming/phd/billboard_seamlessintegration/poison_affine_blending/blend_output/poiss_affine_center/' +self.dst_src_file_name+ '' , bbox_inches = 'tight', pad_inches = 0)


#Main method

if __name__ == "__main__":

    #set1: bundle, set2: kids, set3: denis, set4: stone, set5: tesco, set6: ipad, set7: fuel

    start_time =time.time()
    
    src_image_path ='E:/python_programming/phd/billboard_seamlessintegration/images/foreground/set1/'
    dst_image_path ='E:/python_programming/phd/billboard_seamlessintegration/images/testdataset_background/testdataset3/'
    msk_image_path ='E:/python_programming/phd/billboard_seamlessintegration/images/testdataset_mask/mask_testdataset3/'


    num_of_src_images = len(os.listdir(src_image_path))
    num_of_dst_images = len(os.listdir(dst_image_path))
    num_of_msk_images = len(os.listdir(msk_image_path))

    print('num of src images:', num_of_src_images)
    print('num of dst images:', num_of_dst_images)
    print('num of msk images:', num_of_msk_images)

    read_src_files = natsort.natsorted(glob.glob(src_image_path + "*.jpg"))
    read_dst_files = natsort.natsorted(glob.glob(dst_image_path + "*.jpg"))
    read_msk_files = natsort.natsorted(glob.glob(msk_image_path + "*.jpg"))

    

    src_images = [cv2.imread(file)
                   for file in read_src_files]

    dst_images = [cv2.imread(file)
                   for file in read_dst_files]
    
    msk_images = [cv2.imread(file)
                   for file in read_dst_files]
                  

   

    val = poissaffine(src_images, dst_images, msk_images)

    val.srcresize()
    
    val.fourcorner()

    val.createmask()

    val.poisaffineblending()    
          

    end_time = time.time()

    print(f"Time: {end_time - start_time} seconds")
