#!/usr/bin/env python3

import numpy as np

from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask

import cv2 as cv
import cProfile
import re

DATASET_DIR="../../dataset"

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1

def clean_segmented_image(seg_img):
    # TODO
    # Tip: use either of the two display functions found in utils.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    boxes = np.zeros((0, 4))
    classes = np.zeros((0,1))
    
    return boxes, classes

def remove_dots(seg_img, mask):
    k = mask.shape
    skip_row = np.floor(seg_img.shape[1] / k[1])
    
    i = 0
    while i < seg_img.shape[0] - k[0] + 1:
        j = skip = 0
        while j < seg_img.shape[1] - k[1] + 1:
            window = seg_img[i:i+k[0], j:j+k[1]]
            aggregated_window = window.sum(axis=2)
            masked_aggregated_window = mask * aggregated_window
            if not masked_aggregated_window.any():
                blank = np.zeros((k[0], k[1], seg_img.shape[2]))
                seg_img[i:i+k[0], j:j+k[1]] = blank
                skip += 1
                j += k[1]
            else:
                value = masked_aggregated_window[0, 0]                
                if not (masked_aggregated_window - mask * value).any():
                    skip += 1
                    j += k[1]
                else:
                    j += 1
        if skip == skip_row:
            i += k[0]
        else:
            i += 1
    
    return seg_img

def count_row(image, color, threshold, value, i, j, c, pixels):
    pixels[value] = np.array([i, j])
    value += 1
    
    # Find left-most connected candidate pixel of the same color on next row
    next_j = j + 1
    if c == -1 and next_j < image.shape[1]:
        for k in reversed(range(i - 1, i + 2)):
            if k > 0 and k < image.shape[0]:
                if (image[k, next_j] == color).all():
                    c = k
    
    # Count number of connected pixels on the current row
    next_i = i + 1
    if next_i < image.shape[0] - 1 and (image[next_i, j] == color).all() and value < threshold:
        pixels, value, c = count_row(image, color, threshold, value, next_i, j, c, pixels)
    
    return pixels, value, c

def slow_count_row(image, color, threshold, value, i, j, c, pixels):
    value += 1
    if pixels is None:
        # Initialize the liste
        pixels = np.array([i, j]).reshape((1, 2))
    else:
        # Add current pixel to list
        pixels = np.vstack([pixels, np.array([i, j])])
    
    # Find left-most connected candidate pixel of the same color on next row
    next_j = j + 1
    if c == -1 and next_j < image.shape[1]:
        for k in reversed(range(i - 1, i + 2)):
            if k > 0 and k < image.shape[0]:
                if (image[k, next_j] == color).all():
                    c = k
    
    # Count number of connected pixels on the current row
    next_i = i + 1
    if next_i < image.shape[0] - 1 and (image[next_i, j] == color).all():
        pixels, value, c = count_row(image, color, threshold, value, next_i, j, c, pixels)
    
    return pixels, value, c

def search(image, color, threshold, value, i, j, pixels):
    next_i = i - 1
    if next_i >= 0 and (image[next_i, j] == color).all():
        return search(image, color, threshold, value, next_i, j, pixels)
    else:
        return count(image, color, threshold, value, i, j, pixels)
    
def count(image, color, threshold, value, i, j, pixels):
    if (image[i, j] == color).all():
        # Count number of connected pixels on current row
        pixels, value, c = count_row(image, color, threshold, value, i, j, -1, pixels)

        # Count number of connected pixels on subsequent rows
        if value < threshold and c != -1:
            pixels, value = search(image, color, threshold, value, c, j + 1, pixels)
            
        return pixels, value
    else:
        return None, value

def fast_remove_dots(seg_img):
    dot_colors = np.array([[100, 117, 226],
                            [0, 200, 0],
                            [216, 171, 15],
                            [116, 114, 117]])
    max_size = 40

    for color in dot_colors:
        equal_color = (seg_img == color).all(axis=2)
        indexes = np.where(equal_color)
        for i, j in zip(indexes[0], indexes[1]):
            pixels = np.zeros((max_size, 2)).astype("int")
            pixels, value = count(seg_img, color, max_size, 0, i, j, pixels)

            if value < max_size:
                for i in range(value):
                    seg_img[pixels[i, 0], pixels[i, 1]] = 0
    
    return seg_img

def generate_mask(k):
    # Creates a k x k mask with 1 on the border and 0 inside
    mask = np.ones(k)
    for _ in range(k - 2):
        row = np.zeros(k)
        row[0] += 1
        row[-1] += 1
        mask = np.vstack([mask, row])
    mask = np.vstack([mask, np.ones(k)])

    return mask

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

mask = generate_mask(18)

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        with cProfile.Profile() as pr:
            fast_remove_dots(segmented_obs)

        pr.print_stats()

        segmented_obs = fast_remove_dots(segmented_obs)
        
        # Attenuate the noise
        #segmented_obs = remove_dots(segmented_obs, mask)
        #segmented_obs = cv.medianBlur(segmented_obs, 7)
        #segmented_obs = cv.medianBlur(segmented_obs, 5)
        #segmented_obs = cv.GaussianBlur(segmented_obs,(5,5),0)
        #segmented_obs = cv.medianBlur(segmented_obs, 5)
        
        
        
        display_img_seg_mask(obs, segmented_obs, segmented_obs)
        boxes, classes = clean_segmented_image(segmented_obs)
        # TODO save_npz(obs, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break