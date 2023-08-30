"""Measure test accuracy of models for TRZ"""
import os
import glob
import captcha_solver


DEBUG = False       # if True script will output predictions compared to solutions
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow error messages

TEST_DIRECTORY = os.path.join(os.path.dirname(__file__), f"data{os.sep}test_data")
NETWORK_PATH = os.path.join(os.path.dirname(__file__), f"models{os.sep}model1")

def test():
    print(f"\nMarketplace: TRZ; Network: {NETWORK_PATH}\n")

    total_err = 0
    total = 0

    # List of raw test captchas
    captcha_image_files = glob.glob(f"{TEST_DIRECTORY}{os.sep}*")

    solver = captcha_solver.Solver(network_path=NETWORK_PATH)
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

