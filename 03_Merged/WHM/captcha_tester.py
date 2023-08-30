"""Measure test accuracy of models for WHM"""
import glob
import os
from captcha_solver import Solver

DEBUG = False       # if True script will output predictions compared to solutions
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow error messages

TEST_DIRECTORY = os.path.join(os.path.dirname(__file__), f"data{os.sep}test_data")
NETWORK_PATH = os.path.join(os.path.dirname(__file__), f"models{os.sep}model4")

def test():
    print(f"\nMarketplace: WHM; Network: {NETWORK_PATH}\n")

    total_err = 0
    total = 0

    # List of raw test captchas
    captcha_image_files = glob.glob(f"{TEST_DIRECTORY}{os.sep}*.html")

    solver = Solver(network_path=NETWORK_PATH)
    for captcha in captcha_image_files:
        # Get predicted ids of captcha
        prediction = solver.solve(captcha)

        # Get actual ids of captcha
        with open(captcha.replace(".html", "_sol.txt")) as solution:
            sol_string = solution.readline().strip()
            correct_ids = sol_string.split(",")
            ids = [int(img_id) for img_id in correct_ids]
            ids.sort()

            # Track errors
            if prediction != ids:
                total_err += 1
                print(f"Predicted: {prediction} Correct: {ids}")
            total += 1

        # Debug mode for more information
        if DEBUG:
            print(f"Current filename: {captcha}")
            print(f"Predicted: {prediction} Correct: {ids}")

    print(f"Solved {total-total_err} captchas of {total} correctly. Accuracy = {(total-total_err)/total}")


if __name__ == '__main__':
    test()
