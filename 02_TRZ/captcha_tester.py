"""Measure test accuracy of models for TRZ"""
import os
import glob

import captcha_solver

TEST_DIRECTORY = f"data{os.sep}test_data"

DEBUG = False       # if True script will output predictions compared to solutions
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow error messages

def test():
    total_err = 0
    total = 0

    # List of raw test captchas
    captcha_image_files = glob.glob(f"{TEST_DIRECTORY}{os.sep}*")

    solver = captcha_solver.Solver(model_path="models/model6")
    for img_file in captcha_image_files:
        # Get predicted labels of captcha
        prediction = solver.solve(img_file)

        # Get actual labels of captcha
        filename = os.path.basename(img_file)
        captcha_text = os.path.splitext(filename)[0]
        correct_labels = [char for char in captcha_text]

        # Track errors
        if prediction != correct_labels:
            total_err += 1
            print(f"Predicted: {prediction} Correct: {correct_labels}")
        total += 1

        # Debug mode for more information
        if DEBUG:
            print(f"Predicted: {prediction} Correct: {correct_labels}")

    print(f"Solved {total-total_err} captchas of {total} correctly. Accuracy = {(total-total_err)/total}")



if __name__ == '__main__':
    test()

