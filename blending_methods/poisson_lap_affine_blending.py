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

import scipy.sparse
from scipy.sparse.linalg import spsolve

from creatlapmatrix import laplacian_matrix

import time

#class creation
class poisslapaffine(object):
    
    #class constructor
    def __init__(self, srcimages, dstimages, mskimages):
        self.srcimages = srcimages
        self.dstimages = dstimages
        self.mskimages = mskimages
        

        # Prepare mask and matrix A
        self.y_max, self.x_max = self.dstimages[0].shape[:-1]
        self.y_min, self.x_min = 0, 0
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min

        print('ymax:', self.y_max, '|', 'ymin:', self.y_min)
        print('xmax:', self.x_max, '|', 'xmin:', self.x_min)
        print('x_range:', self.x_range)
        print('y_range:', self.y_range)

        
    #method to resize
    def srcresize(self):

        #self.src_resized_array = []
        

        print('src shape:', self.srcimages[0].shape)
        print('dst shape:', self.dstimages[0].shape)
        print('msk shape:', self.mskimages[0].shape)

        #for i in range(len(self.srcimages)):
        self.im_src_resized = cv2.resize(self.srcimages[0], (int(self.dstimages[0].shape[1]), int(self.dstimages[0].shape[0])))

        #self.src_resized_array.append(self.im_src_resized)

    #method for four corner
    def fourcorner(self):

        df = pd.read_csv(r'E:/python_programming/phd/billboard_seamlessintegration/csv_data/data4corner_set4.csv')

        self.im_out_array = []

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

        for i in range(len(self.dstimages)):


            #lt, lb, rt, rb
            #np.array([[0, 0], [0, 512], [512, 0], [512, 512]]) #tesco, denos, stone advert
            #np.array([[34, 125], [34, 323], [491, 125], [491, 323]]) #scene_12 512 x 512 image size
            #np.array([[159, 95],[159, 191],[362, 95],[362, 191]]) #scene_52 #512 x 512 image size
            #np.array([[96, 32],[96, 146],[352, 32],[352, 146]]) # scene_256 #512 x 512 image size

            #read four corner data from csv file
            #Four corners of the advert in source image
            pts_src = np.array([[96, 32],[96, 146],[352, 32],[352, 146]]) 
            #print('src image coordinates:', pts_src)

            #Four corners of the billboard in destination image.
            pts_dst = np.array([[x1[i], y1[i]], [x3[i], y3[i]] ,[x2[i], y2[i]], [x4[i], y4[i]]])

            self.pts_imagename = imagename[i]

            #print(pts_dst)
            #left-top, left-bottom,  right-top, right-bottom
            offset = (0, 66)

            #Calculate Homography
            h, status = cv2.findHomography(pts_src, pts_dst)
            #print(h)
            #print(status)

            #Warp source image to destination based on homography
            self.im_out = cv2.warpPerspective(self.im_src_resized, h, (self.dstimages[i].shape[1], self.dstimages[i].shape[0]))
            
        
            #print('image out size:', self.im_out.shape[:-1])

            src_file_name = read_src_files[0]
            src_image_name = os.path.basename(src_file_name)[0]

            self.pts_filename = src_image_name + self.pts_imagename

            print('im out imagename:', self.pts_filename)
            
            plt.axis('off')
            plt.imshow(self.im_out[:,:,::-1])
            #plt.savefig('E:/python_programming/phd/billboard_seamlessintegration/poisson_lap_affine_blending/blend_output/affine_output/' +self.pts_filename+ '', bbox_inches = 'tight', pad_inches = 0)
            #plt.show()
            
            self.im_out_array.append(self.im_out)
        
            


    #method for blending
    def  poislapaffineblending(self):

        self.blended_array = []
        
        
        
        ########### filename ########################
        for i in range(len(self.dstimages)):


            src_file_name = read_src_files[0]
            src_image_name = os.path.basename(src_file_name)

            dst_file_name = read_dst_files[i]
            dst_image_name = os.path.splitext(os.path.basename(dst_file_name))[0]

            dst_src_file_name = dst_image_name+'_' + src_image_name 
            print('blended filename:',dst_src_file_name)

            ############### generate maskgray ##################

            self.maskgray = cv2.cvtColor(self.mskimages[i], cv2.COLOR_BGR2GRAY)            

            self.maskgray = self.maskgray[self.y_min:self.y_max, self.x_min:self.x_max]
            self.maskgray[self.maskgray != 0] = 1
            print('maskgray:', self.maskgray)

            #create matrix 
            self.mat_D = scipy.sparse.lil_matrix((512, 512))
            self.mat_D.setdiag(-1, -1)
            self.mat_D.setdiag(-4)
            self.mat_D.setdiag(-1, 1)

            self.mat_A = scipy.sparse.block_diag([self.mat_D] * 512).tolil()

            self.mat_A.setdiag(-1, 1 * 512)
            self.mat_A.setdiag(-1, -1 * 512)

            self.laplacian = self.mat_A.tocsc()#converting matrix to compressed sparse column
            
            for y in range(1, self.y_range - 1):
                for x in range(1, self.x_range - 1):
                    if self.maskgray[y, x] == 0:
                        k = x + y * self.x_range
                        self.mat_A[k, k] = 1
                        self.mat_A[k, k + 1] = 0
                        self.mat_A[k, k - 1] = 0
                        self.mat_A[k, k + self.x_range] = 0
                        self.mat_A[k, k - self.x_range] = 0
            self.mat_A = self.mat_A.tocsc()

            self.mask_flat = self.maskgray.flatten() #flatten converting multi dimensional array  into one dimension array

            print('mask_flat:',self.mask_flat)

            #####blending code begins here#########

            #print('imoutshape2:',self.im_out.shape[2])#3
            #print('imoutarray:', len(self.im_out_array))

            start_time = time.time()
            for channel in range(self.im_out_array[i].shape[2]):

               #start_time = time.time()
                
               self.source_flat = self.im_out_array[i][self.y_min:self.y_max, self.x_min:self.x_max, channel].flatten()
               self.target_flat = self.dstimages[i][self.y_min:self.y_max, self.x_min:self.x_max, channel].flatten()    
               
        

                # inside the mask:
                # \Delta f = div v = \Delta g       
               alpha = 0.9
               self.mat_b = self.laplacian.dot(self.source_flat)*alpha

                # outside the mask:
                # f = t
               self.mat_b[self.mask_flat == 0] = self.target_flat[self.mask_flat == 0]
    
               x = spsolve(self.mat_A, self.mat_b)    
      
               x = x.reshape((self.y_range, self.x_range))
    
               x[x > 255] = 255
               x[x < 0] = 0
    
               x = x.astype('uint8')
    
               self.dstimages[i][self.y_min:self.y_max, self.x_min:self.x_max, channel] = x

            end_time = time.time()

            print(f"Time: {end_time - start_time} seconds")

            
               #print('channel:', channel)
               #print('source flat:', self.source_flat)
               #print('target flat:', self.target_flat)
               #print('mask flat:', self.mask_flat)
               #print('x:',x)
               #print('matb:', self.mat_b)

            #plt.imshow(self.im_out_array[i][:,:,::-1])

            plt.imshow(self.dstimages[i][:,:,::-1])
            
            plt.axis('off')
            #plt.savefig('E:/python_programming/phd/billboard_seamlessintegration/poisson_lap_affine_blending/blend_output/pois_lap_output_f/' +dst_src_file_name+ '', bbox_inches = 'tight', pad_inches = 0)
            #plt.show()




#Main method

if __name__ == "__main__":

    #set1: bundle, set2: kid, set3: denis, set4: stone, set5: tesco, set7: fuel

    
    
    src_image_path ='E:/python_programming/phd/billboard_seamlessintegration/images/foreground/set7/'
    dst_image_path ='E:/python_programming/phd/billboard_seamlessintegration/images/testdataset_background/testdataset4/'
    mask_image_path = 'E:/python_programming/phd/billboard_seamlessintegration/images/testdataset_mask/mask_testdataset4/' #mask of background image


    num_of_src_images = len(os.listdir(src_image_path))
    num_of_dst_images = len(os.listdir(dst_image_path))
    num_of_msk_images = len(os.listdir(mask_image_path))

    print('num of src images:', num_of_src_images)
    print('num of dst images:', num_of_dst_images)
    print('num of msk images:', num_of_msk_images)

    read_src_files = natsort.natsorted(glob.glob(src_image_path + "*.jpg"))
    read_dst_files = natsort.natsorted(glob.glob(dst_image_path + "*.jpg"))
    read_msk_files =  natsort.natsorted(glob.glob(mask_image_path + "*.jpg"))

    

    src_images = [cv2.imread(file)
                   for file in read_src_files ]

    dst_images = [cv2.imread(file)
                   for file in read_dst_files]#[skimage.io.imread(file)for file in read_dst_files]

    msk_images = [cv2.imread(file)
                   for file in read_msk_files]

    #generate the matrix A
    #mat_A = laplacian_matrix(512, 512)
    #laplacian = mat_A.tocsc() #converting to matrix to compressed sparese column

    val = poisslapaffine(src_images, dst_images, msk_images)
    
    val.srcresize()

    val.fourcorner()
    
    val.poislapaffineblending()

    
