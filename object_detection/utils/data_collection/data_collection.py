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
    boxes = np.zeros((0, 4))
    classes = np.zeros((0,1))

    for i, color in enumerate(class_colors):
        mask = (seg_img == color).all(axis=2)
        image = np.where(mask, 1, 0).astype(np.uint8)
        
        contours, _ = cv.findContours(image, 1, 2)

        for contour in contours:
            if cv.contourArea(contour) > 25:
                x, y, w, h = cv.boundingRect(contour)
                x2 = x + w
                y2 = y + h
                
                if boxes.shape[0] == 0 or not np.bitwise_and(
                    np.bitwise_and(
                        x >= boxes[:, 0],
                        y >= boxes[:, 1]
                    ),
                    np.bitwise_and(
                        x2 <= boxes[:, 2],
                        y2 <= boxes[:, 3]
                    )
                ).any():
                    boxes = np.vstack([boxes, [x, y, x2, y2]])
                    classes = np.vstack([classes, [i + 1]])

                    if INTERACTIVE:
                        cv.rectangle(seg_img, (x, y), (x2, y2), (15, 171, 216), 1)

    return boxes, classes
    
seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

class_colors = np.array(
    [
        [100, 117, 226],
        [226, 111, 101],
        [116, 114, 117],
        [216, 171,  15]
    ],
    dtype=np.uint8
)

# Session config
RENDER = False
INTERACTIVE = False

while True:
    obs = environment.reset()
    #environment.render(segment=True)

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, _, done, _ = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        #environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        image = cv.resize(segmented_obs, (224, 224))
        
        #display_img_seg_mask(obs, image, image)
        boxes, classes = clean_segmented_image(image)
        save_npz(image, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break