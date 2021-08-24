# Author: aqeelanwar
# Created: 6 July,2020, 12:14 AM
# Email: aqeel.anwar@gatech.edu

from PIL import ImageColor
import cv2
import numpy as np

COLOR = [
    "#fc1c1a",
    "#177ABC",
    "#94B6D2",
    "#A5AB81",
    "#DD8047",
    "#6b425e",
    "#e26d5a",
    "#c92c48",
    "#6a506d",
    "#ffc900",
    "#ffffff",
    "#000000",
    "#49ff00",
]


def color_the_mask(mask_image, color, intensity):
    assert 0 <= intensity <= 1, "intensity should be between 0 and 1"
    RGB_color = ImageColor.getcolor(color, "RGB")
    RGB_color = (RGB_color[2], RGB_color[1], RGB_color[0])
    orig_shape = mask_image.shape
    bit_mask = mask_image[:, :, 3]
    mask_image = mask_image[:, :, 0:3]

    color_image = np.full(mask_image.shape, RGB_color, np.uint8)
    mask_color = cv2.addWeighted(mask_image, 1 - intensity, color_image, intensity, 0)
    mask_color = cv2.bitwise_and(mask_color, mask_color, mask=bit_mask)
    colored_mask = np.zeros(orig_shape, dtype=np.uint8)
    colored_mask[:, :, 0:3] = mask_color
    colored_mask[:, :, 3] = bit_mask
    return colored_mask


def texture_the_mask(mask_image, texture_path, intensity):
    assert 0 <= intensity <= 1, "intensity should be between 0 and 1"
    orig_shape = mask_image.shape
    bit_mask = mask_image[:, :, 3]
    mask_image = mask_image[:, :, 0:3]
    texture_image = cv2.imread(texture_path)
    texture_image = cv2.resize(texture_image, (orig_shape[1], orig_shape[0]))

    mask_texture = cv2.addWeighted(
        mask_image, 1 - intensity, texture_image, intensity, 0
    )
    mask_texture = cv2.bitwise_and(mask_texture, mask_texture, mask=bit_mask)
    textured_mask = np.zeros(orig_shape, dtype=np.uint8)
    textured_mask[:, :, 0:3] = mask_texture
    textured_mask[:, :, 3] = bit_mask

    return textured_mask
