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

import cv2
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":
    
    images_path = "data/"
    image_name = "cathedral.jpg"
    
    # read in the image
    img = cv2.imread(images_path + image_name)
    
    H, W, C = img.shape
    
    print("Image shape: ", img.shape)
    
    # convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # split the image into 3 parts by height 
    B, G, R = gray_img[:H//3, :], gray_img[H//3:2*H//3, :], gray_img[2*H//3:, :]
    # display the 3 parts in a single window side by side usinf matplotlib subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(R, cmap='gray')
    ax[0].set_title('R')
    ax[1].imshow(G, cmap='gray')
    ax[1].set_title('G')
    ax[2].imshow(B, cmap='gray')
    ax[2].set_title('B')
    plt.show()
    
    # Make sure all the 3 parts have the same shape
    # If not, pad the smaller ones with zeros to make them the same size
    # RGB -> BGR
    if B.shape != G.shape:
        B = np.pad(B, ((0, G.shape[0]-B.shape[0]), (0, 0)), 'constant')
    if B.shape != R.shape:
        B = np.pad(B, ((0, R.shape[0]-B.shape[0]), (0, 0)), 'constant')
    if G.shape != R.shape:
        G = np.pad(G, ((0, R.shape[0]-G.shape[0]), (0, 0)), 'constant')
        
    print("R shape: %s, G shape: %s, B shape: %s" % (R.shape, G.shape, B.shape))
    
    recon_image_shape = R.shape
    
    # create a new image to store the reconstructed image
    img_reconstruct = np.zeros((recon_image_shape[0], recon_image_shape[1], 3), dtype=np.uint8)
    print("Reconstructed Image shape: ", img_reconstruct.shape)
    
    # img_reconstruct = np.zeros(recon_image_shape, dtype=np.uint8)
    
    
    # over lay the 3 parts on top of each other
    print("Reconstructed Image shape: ", img_reconstruct.shape)
    img_reconstruct[:,:,0] = R
    img_reconstruct[:,:,1] = G
    img_reconstruct[:,:,2] = B
    
    # display the reconstructed image
    plt.imshow(img_reconstruct)
    plt.title('Reconstructed Image')
    plt.show()
    
    
    
        
     
    