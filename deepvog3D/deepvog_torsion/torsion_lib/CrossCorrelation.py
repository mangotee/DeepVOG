import numpy as np
import cv2
from cv2 import GaussianBlur
from skimage.exposure import equalize_adapthist as adhist
from .PolarTransform import polarTransform
from scipy.signal import correlate
from skimage import img_as_float
from skimage.transform import resize

import os

def guassianMap(img_mean, img_std, useful_map, filter_sigma = 1):
    guassian_map = np.random.normal(img_mean, img_std, useful_map.shape)
    guassian_map[guassian_map < 0] = 0
    guassian_map[guassian_map > 1] = 1
    guassian_map = GaussianBlur(guassian_map, ksize= (0,0), sigmaX = filter_sigma)
    guassian_map[useful_map==1] = 0
    return guassian_map

def normalizeRobust(img, percentiles=[5,95], bg_threshold=0.05):
    # find intensity percentiles
    valid_pixel_intensities = img[img>bg_threshold]
    th_low, th_high = np.percentile(valid_pixel_intensities,percentiles)
    # rescale intensities
    img -= th_low
    img /= (th_high-th_low)
    # clip
    img = np.clip(img,0.0,1.0)
    return img
    
def genPolar(img, useful_map, center, template=False, filter_sigma = 1, adhist_times = 2, apply_gaussian_noise=True, angular_resolution=0.02):
    assert img.shape==useful_map.shape # img must have same shape as network predictions
    assert img.max()<=1.0  # img must be normalized
    
    if adhist_times >= 1:
        img_enhanced = img_as_float(adhist(img))
    else:
        img_enhanced = img_as_float(img)
        
    img_enhanced[useful_map == 0] = 0
    
    # If no radial/tangential filtering is performed, alter the codes to contain only one polarTransform function to speed up performance
    output_img, r, theta = polarTransform(img_enhanced, np.where(useful_map==1)[::-1], origin=center, angular_resolution=angular_resolution )
    
    # adaptive histogram equalization (CLAHE)
    if True:
        if adhist_times >= 2:
            kernel_size = None
            if (output_img.shape[0] < 8): # solving the "Division by zero" error in adhist function (kernel_size = 0 if img.shape[?] < 8)
                kernel_size = [1,1]
                if (output_img.shape[1] > 8):
                    kernel_size[1] = output_img.shape[1]//8
            output_img = adhist(output_img, kernel_size)
    else:
        # if CLAHE it too expensive, do percentile normalization instead
        bg_threshold = 0.25
        output_img_bg_mask = output_img <= bg_threshold
        output_img = normalizeRobust(output_img, bg_threshold=bg_threshold)
    
    # compute features of polar transform (i.e. quality measures, which we can use for template updating)
    polar_coverage_region = output_img>0.01
    polar_coverage_percent = np.sum(polar_coverage_region) / np.prod(output_img.shape)
    polar_xgradient = np.abs(np.diff(output_img,axis=1))
    polar_xgradient_sum = np.sum(polar_xgradient[polar_coverage_region[:,1:]])
    polar_xgradient_average = np.sum( polar_xgradient[polar_coverage_region[:,1:]] / np.sum(polar_coverage_region[:,1:]) )
    
    quality_measures = dict({'polar_coverage_percent': polar_coverage_percent,
                             'polar_xgradient_sum': polar_xgradient_sum,
                             'polar_xgradient_average': polar_xgradient_average})
    
    if apply_gaussian_noise:
        # create a map with gaussian noise to fill the black regions after polar transform
        guassian_map = guassianMap(0.5, 0.2, useful_map, filter_sigma = filter_sigma)# polar transform of gaussian noise map
        output_gaussian, r_gaussian, theta_gaussian = polarTransform(guassian_map, 
                                                                     np.where(useful_map==1)[::-1], 
                                                                     origin=center,
                                                                     angular_resolution=angular_resolution )
        # additive noise onto iris map
        output = output_img + output_gaussian
    else:
        # instead of filling the bg with smooth noise, fill it simply with the average iris intensity
        bg_threshold = 0.25
        output_img_bg_mask = output_img <= bg_threshold
        output_img = normalizeRobust(output_img, bg_threshold=bg_threshold)
        output = output_img
        output[output_img_bg_mask] = np.mean(output_img[np.logical_not(output_img_bg_mask)])

    if template == True:
        extra_index, extra_rad = int(25*(1/angular_resolution)), np.deg2rad(25)
        output_longer = np.concatenate((output[:,output.shape[1]-extra_index:], output, output[:, 0:extra_index]), axis = 1)
        return output, output_longer, r, theta, extra_rad, quality_measures
    else:
        return output, r, theta, quality_measures



def findTorsion(output_template, img_r, useful_map_r, center,  filter_sigma = 1, adhist_times = 2, angular_resolution=0.02):
    # polar transform img_r to output_r
    output_r, r_r, theta_r, quality_measures = genPolar(img_r, 
                                                        useful_map_r, 
                                                        center , 
                                                        template = False,
                                                        filter_sigma = filter_sigma, 
                                                        adhist_times = adhist_times,
                                                        apply_gaussian_noise = False,
                                                        angular_resolution=angular_resolution)
    
    # template matching
    cols = output_r.shape[1]
    
    #interp = cv2.INTER_CUBIC
    interp = cv2.INTER_LINEAR
    output_r = cv2.resize(output_r, (cols, output_template.shape[0]), interpolation=interp)
    
    output_r_pad = np.pad(output_r, ((0, 0),(cols//2, cols//2)),'constant',constant_values=(0,0))

    output_r_pad = output_r_pad.astype(np.float32)
    output_template = output_template.astype(np.float32)

    corr = cv2.matchTemplate(output_r_pad, output_template, cv2.TM_CCORR_NORMED)
    
    max_index = np.argmax(corr)
    rotation = max_index*angular_resolution-(180-25)
    #print(rotation)
    
    return rotation.squeeze(), (output_r, r_r, theta_r), corr, quality_measures