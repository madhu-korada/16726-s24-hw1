# -*-coding:utf-8 -*-
'''
@File    :   main_hw1.py
@Time    :   2024/01/30 14:49:34
@Author  :   Madhu Korada 
@Version :   1.0
@Contact :   mkorada@cs.cmu.edu
@License :   (C)Copyright 2024-2025, Madhu Korada
@Desc    :   None
'''
import os
import cv2
import time
import skimage
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt

def ssd(image1, image2):
    """
    This function computes the sum of squared differences between two images
    
    @param image1: The first image
    @param image2: The second image
    
    @return: The sum of squared differences between the two images
    """
    return np.sum((image1 - image2) ** 2)

def ncc(image1, image2):
    """
    This function computes the normalized cross-correlation between two images
    
    @param image1: The first image
    @param image2: The second image
    
    @return: The normalized cross-correlation between the two images
    """
    # Normalize the images
    norm_image1 = image1 / np.linalg.norm(image1)
    norm_image2 = image2 / np.linalg.norm(image2)
    
    # Compute the dot product
    return np.sum(norm_image1 * norm_image2)

def align_images(image1, image2, displacement_rangem, use_ssd=True):
    """
    This function aligns two images by finding the displacement that minimizes the sum of squared differences
    
    @param image1: The first image
    @param image2: The second image
    @param displacement_range: The maximum displacement to search for
    
    @return: The displacement that minimizes the sum of squared differences
    """
    min_ssd = float('inf')
    max_ncc = float('-inf')
    best_displacement = (0, 0)
    
    for i in range(-displacement_range, displacement_range + 1):
        for j in range(-displacement_range, displacement_range + 1):
            displaced_image = np.roll(image2, shift=(i, j), axis=(0, 1))
            if use_ssd:
                current_ssd = ssd(image1, displaced_image)
                if current_ssd < min_ssd:
                    min_ssd = current_ssd
                    best_displacement = (i, j)
            else:
                current_ncc = ncc(image1, displaced_image)
                if current_ncc > max_ncc:
                    max_ncc = current_ncc
                    best_displacement = (i, j)
    return best_displacement

def align_images_opt(image1, image2, prev_displacement, displacement_range=15, use_ssd=True):
    """
    This function aligns two images by finding the displacement that maximizes the normalized cross-correlation or minimizes the sum of squared differences
    
    @param image1: The first image
    @param image2: The second image
    @param prev_displacement: The previous displacement
    @param displacement_range: The maximum displacement to search for
    
    @return: The displacement that maximizes the normalized cross-correlation or minimizes the sum of squared differences
    """
    dx, dy = prev_displacement[1], prev_displacement[0]
    best_displacement = np.array([0, 0])
    max_ncc = float('-inf')
    loss_min = float('inf')
    # print("Level:", level, "Displacement:", best_displacement, 
    #       "Displacement Range:", displacement_range, "dx:", dx, "dy:", dy)
    # print("Itr Range:", -displacement_range, displacement_range + 1)
    for i in range(dy - displacement_range, dy + displacement_range + 1):
        for j in range(dx - displacement_range, dx + displacement_range + 1):
            displaced_image = np.roll(image1, shift=(i, j), axis=(0, 1))
            if use_ssd:
                current_ssd = ssd(displaced_image, image2)
                if current_ssd < loss_min:
                    loss_min = current_ssd
                    best_displacement = np.array([i, j])
            else:
                current_ncc = ncc(displaced_image, image2)
                if current_ncc > max_ncc:
                    max_ncc = current_ncc
                    best_displacement = np.array([i, j])
    return best_displacement


# pyramid
def pyramid_align(target, ref, displacement_range=15, no_levels=None, downsample_factor=2):
    """
    This function aligns two images by finding the displacement that maximizes the normalized cross-correlation or minimizes the sum of squared differences
    
    @param target: The first image
    @param ref: The second image
    @param displacement_range: The maximum displacement to search for
    @param no_levels: The number of levels in the pyramid
    @param downsample_factor: The factor by which to downsample the images
    
    @return: The displacement that maximizes the normalized cross-correlation or minimizes the sum of squared differences
    """
    
    best_displacement = np.array([0, 0]) 

    # Calculate the number of levels in the pyramid
    if no_levels is None:
        no_levels = int(np.log(target.shape[0])/np.log(downsample_factor)) # + 1

    if (no_levels < 9):
        best_displacement = align_images_opt(target, ref, best_displacement, displacement_range, use_ssd=True)
        print("image is small, not using pyramid")
        return best_displacement

    # print("The pyramid of downsampling has " + str(no_levels + 1) + " layers") 
    # Do the pyramid alignment starting from the top of the pyramid
    for scale in range(no_levels - 7, -1, -1):
        target_scaled = cv2.resize(target, (0, 0), fx = 1/(downsample_factor**scale), fy = 1/(downsample_factor**scale))
        ref_scaled = cv2.resize(ref, (0, 0), fx = 1/(downsample_factor**scale), fy = 1/(downsample_factor**scale))
        
        best_displacement = align_images_opt(target_scaled, ref_scaled, best_displacement, displacement_range=15, use_ssd=True)
        # print("best_displacement:", best_displacement)
        # displacement_range = displacement_range // 2 # reduce the displacement range by half
        displacement_range = displacement_range - 2
        best_displacement = best_displacement * downsample_factor

    # Adjust the displacement to the original image size
    best_displacement = best_displacement // downsample_factor
    return best_displacement

def plot_rgb_images(R, G, B):
    """
    This function plots the 3 parts of the image side by side
    
    @param R: The red part of the image
    @param G: The green part of the image
    @param B: The blue part of the image
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(R, cmap='gray')
    ax[0].set_title('R')
    ax[1].imshow(G, cmap='gray')
    ax[1].set_title('G')
    ax[2].imshow(B, cmap='gray')
    ax[2].set_title('B')
    plt.show()

def same_size(R, G, B):
    """
    This function makes sure that all the 3 parts of the image have the same size
    
    @param R: The red part of the image
    @param G: The green part of the image
    @param B: The blue part of the image
    
    @return: The 3 parts of the image with the same size
    """
    if B.shape != G.shape:
        B = np.pad(B, ((0, G.shape[0]-B.shape[0]), (0, 0)), 'constant')
    if B.shape != R.shape:
        B = np.pad(B, ((0, R.shape[0]-B.shape[0]), (0, 0)), 'constant')
    if G.shape != R.shape:
        G = np.pad(G, ((0, R.shape[0]-G.shape[0]), (0, 0)), 'constant')
    return R, G, B


if __name__ == "__main__":
    
    images_path = "data/"
    output_path = "output/"
    plot_output = False
    plot_rgb = False
    
    for image_name in os.listdir(images_path):
        print("Processing Image: ", image_name.replace(".tif", ""))
        img = cv2.imread(images_path + image_name)
        H, W, C = img.shape
        displacement_range = img.shape[1] // 50
        
        # convert the image to grayscale and split the image into 3 parts by height 
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        B, G, R = gray_img[:H//3, :], gray_img[H//3:2*H//3, :], gray_img[2*H//3:, :]
                
        # Make sure all the 3 parts have the same shape
        R, G, B = same_size(R, G, B)
        if plot_rgb:
            plot_rgb_images(R, G, B)
        
        start = time.time()
        best_displacement = pyramid_align(G, B)
        G = np.roll(G, shift=best_displacement, axis=(0, 1))
        print("Aligning Green to Blue with displacement: ", best_displacement)
        
        best_displacement = pyramid_align(R, B)
        R = np.roll(R, shift=best_displacement, axis=(0, 1))
        print("Aligning Red to Blue with displacement: ", best_displacement)
        print("The time of execution of above program is :", time.time() - start)
        
        recon_image_shape = R.shape
        # create a new image to store the reconstructed image
        img_reconstruct = np.zeros((recon_image_shape[0], recon_image_shape[1], 3), dtype=np.uint8)
        print("Reconstructed Image shape: ", img_reconstruct.shape)
        
        # over lay the 3 parts on top of each other
        img_reconstruct[:,:,0] = R
        img_reconstruct[:,:,1] = G
        img_reconstruct[:,:,2] = B
        
        # display the reconstructed image
        if plot_output:
            plt.imshow(img_reconstruct)
            plt.title('Reconstructed Image')
            plt.show()

        # downsample the image if the size is greater than 1000
        if (img_reconstruct.shape[0] > 1000) or (img_reconstruct.shape[1] > 1000):
            scale = int(np.floor(np.log2(min(img_reconstruct.shape)) - np.log2(512))) + 6
            img_reconstruct = cv2.resize(img_reconstruct, (0, 0), fx=2**scale, fy=2**scale)
        
        print("Resized reconstructed Image shape: ", img_reconstruct.shape)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        out_image_path = output_path + "reconstructed_" + image_name.replace(".tif", ".jpg")
        img_reconstruct = skimage.img_as_ubyte(img_reconstruct)
        print("Saving the reconstructed image to: ", out_image_path)
        skio.imsave(out_image_path, img_reconstruct)
        print("\n\n")
        
        # # save the reconstructed image
        # img_reconstruct = cv2.cvtColor(img_reconstruct, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(out_image_path, img_reconstruct)
        
