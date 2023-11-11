# Import all the libraries (unchanged)


import collections
from itertools import chain
import urllib.request as request
import pickle

import numpy as np

import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize

import matplotlib.pyplot as plt

import skimage.io
import skimage.transform

import os
import cv2
import natsort
import glob

from libsvm.svmutil import *
from libsvm import svmutil

import csv
#import xlsx

#natural Scene Statistics in the Spatial Domain
def normalize_kernel(kernel):
    return kernel / np.sum(kernel)

def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(gaussian_kernel)

def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')

def local_deviation(image, local_mean, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))


def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):
    C = 1/255
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(image, kernel, 'same')# performs a 2-dimensional convolution operation on input arrays.
    local_var = local_deviation(image, local_mean, kernel)

    return (image - local_mean) / (local_var + C)


def generalized_gaussian_dist(x, alpha, sigma):
    beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))

    coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
    return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)


#Pairwise products of neighboring MSCN coefficients
def calculate_pair_product_coefficients(mscn_coefficients):
    return collections.OrderedDict({
        'mscn': mscn_coefficients,
        'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
        'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
        'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
        'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    })


def asymmetric_generalized_gaussian(x, nu, sigma_l, sigma_r):
    def beta(sigma):
        return sigma * np.sqrt(special.gamma(1 / nu) / special.gamma(3 / nu))

    coefficient = nu / ((beta(sigma_l) + beta(sigma_r)) * special.gamma(1 / nu))
    f = lambda x, sigma: coefficient * np.exp(-(x / beta(sigma)) ** nu)

    return np.where(x < 0, f(-x, sigma_l), f(x, sigma_r))


#Fitting Asymmetric Generalized Gaussian Distribution

def asymmetric_generalized_gaussian_fit(x):
    def estimate_phi(alpha):
        numerator = special.gamma(2 / alpha) ** 2
        denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
        return numerator / denominator

    def estimate_r_hat(x):
        size = np.prod(x.shape)
        return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

    def estimate_R_hat(r_hat, gamma):
        numerator = (gamma ** 3 + 1) * (gamma + 1)
        denominator = (gamma ** 2 + 1) ** 2
        return r_hat * numerator / denominator

    def mean_squares_sum(x, filter = lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = np.sum(filtered_values ** 2)
        return squares_sum / ((filtered_values.shape))

    def estimate_gamma(x):
        left_squares = mean_squares_sum(x, lambda z: z < 0)
        right_squares = mean_squares_sum(x, lambda z: z >= 0)

        return np.sqrt(left_squares) / np.sqrt(right_squares)

    def estimate_alpha(x):
        r_hat = estimate_r_hat(x)
        gamma = estimate_gamma(x)
        R_hat = estimate_R_hat(r_hat, gamma)

        solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x

        return solution[0]

    def estimate_sigma(x, alpha, filter = lambda z: z < 0):
        return np.sqrt(mean_squares_sum(x, filter))

    def estimate_mean(alpha, sigma_l, sigma_r):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))

    alpha = estimate_alpha(x)
    sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
    sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)

    constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    mean = estimate_mean(alpha, sigma_l, sigma_r)

    return alpha, mean, sigma_l, sigma_r



#Calculate BRISQUE features

def calculate_brisque_features(image, kernel_size=7, sigma=7/6):
    def calculate_features(coefficients_name, coefficients, accum=np.array([])):
        alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)

        if coefficients_name == 'mscn':
            var = (sigma_l ** 2 + sigma_r ** 2) / 2
            return [alpha, var]

        return [alpha, mean, sigma_l ** 2, sigma_r ** 2]

    mscn_coefficients = calculate_mscn_coefficients(image, kernel_size, sigma)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)

    features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
    #print('features:', features)

    flatten_features = list(chain.from_iterable(features))
    #print('flatten_features:', flatten_features)

    flattenfeature_array = np.array(flatten_features)

    #print('flattenfeature_array:', flattenfeature_array)

    return flattenfeature_array

def process_image_quality(image_path):
    image = skimage.io.imread(image_path)
    #rgb_image = skimage.color.rgba2rgb(image) #only for alpha affine for 4 channel images
    gray_image = skimage.color.rgb2gray(image) #skimage.color.rgb2gray(rgb_image)

    # Calculate Coefficients
    mscn_coefficients = calculate_mscn_coefficients(gray_image, 7, 7/6)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)

    # ... (rest of the code for plotting histograms, fitting, etc. unchanged)

    plt.rcParams["figure.figsize"] = 12, 11
    '''
    for name, coeff in coefficients.items():
        plot_histogram(coeff.ravel(), name)

    plt.axis([-2.5, 2.5, 0, 1.05])
    plt.xlabel('MSCN')
    plt.ylabel('Number of coefficicients')
    plt.legend()
    plt.show()
    '''
    brisque_features = calculate_brisque_features(gray_image, kernel_size=7, sigma=7/6)

    # Resize Image and Calculate BRISQUE Features
    downscaled_image = cv2.resize(gray_image, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC)
    downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)
    brisque_features = np.concatenate((brisque_features, downscale_brisque_features))

    return brisque_features

def scale_features(features):
    with open('/python_programming/phd/billboard_integration/image quality assessment/normalize.pickle', 'rb') as handle:
        scale_params = pickle.load(handle)

    min_ = np.array(scale_params['min_'])
    max_ = np.array(scale_params['max_'])

    #print('min_:', min_)
    #print('max_:', max_)

    minmax = -1 + (2.0 / (max_ - min_) * (features - min_))

    #print('minmax:', minmax)

    return minmax


# Trained with LIVE IQA database, 29 reference images, 779 distorted images
def calculate_image_quality_score(brisque_features):
    model = svmutil.svm_load_model('/python_programming/phd/billboard_integration/image quality assessment/brisque_svm.txt')
    scaled_brisque_features = scale_features(brisque_features)

    x, idx = svmutil.gen_svm_nodearray(
        scaled_brisque_features,
        isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))

    nr_classifier = 1
    prob_estimates = (svmutil.c_double * nr_classifier)()

    return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)


# Process multiple images
# Assuming all your images are in a folder 
image_folder = 'E:/python_programming/phd/billboard_seamlessintegration/poisson_lap_affine_blending/blend_output/pois_lap_output_all'
image_paths = [f"{image_folder}/{filename}" for filename in os.listdir(image_folder)]
with open('E:/python_programming/phd/billboard_seamlessintegration/brisque_score/brisque_csv_file/poislap_affine/poislap_brisque_all.csv','w', newline = '') as f:
        writer = csv.writer(f)#create the csv writer
        header = ['image_name', 'score']
        writer.writerow(header)# write a row to the csv file
#

# Now process each image and calculate BRISQUE score
for image_path in natsort.natsorted(image_paths):
    imagefilename = image_path
    imagename = os.path.basename(imagefilename)
    #print('imagename:', imagename)
    brisque_features = process_image_quality(image_path)
    score = calculate_image_quality_score(brisque_features)
    print(f"Image: {imagename}, BRISQUE Score: {score}")
    
    with open('E:/python_programming/phd/billboard_seamlessintegration/brisque_score/brisque_csv_file/poislap_affine/poislap_brisque_all.csv','a', newline = '') as f:
        writer = csv.writer(f)#create the csv writer
        data = [imagename, score]
        writer.writerow(data)# write a row to the csv file
    

