"""Data preprocessing for TRZ CAPTCHAs"""
import os
import glob
import cv2
import shutil

import image_splitter

BASE_DIR = os.path.join(os.path.dirname(__file__), f"data{os.sep}raw_training_data")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), f"data{os.sep}training_data")

def create():

    # List of raw training captchas
    captcha_image_files = glob.glob(f"{BASE_DIR}{os.sep}*")

    # Count for each key
    counts = {}

    # (Re-)create folder if existing
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.mkdir(OUTPUT_FOLDER)

    # Iterate over training images
    for file in captcha_image_files:
        # Get list of single letters
        letter_images = image_splitter.split(file)

        # If image couldn't be split properly, skip image instead of bad training data
        if letter_images is None:
            continue

        # Filename containing captcha text
        filename = os.path.basename(file)
        captcha_text = os.path.splitext(filename)[0]

        for letter_image, letter in zip(letter_images, captcha_text):
            # Create target folder if not existing
            target_path = os.path.join(OUTPUT_FOLDER, letter)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            # Save letter image with unique name
            count = counts.get(letter, 1)
            p = os.path.join(target_path, f"{str(count).zfill(6)}.png")
            cv2.imwrite(p, letter_image)

            # Increment count for current key
            counts[letter] = count + 1


if __name__ == '__main__':
    create()


