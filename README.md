# Bachelor's Thesis

---

## Comparing Machine Learning Models for Breaking CAPTCHAs in Darknet Marketplaces

---

This thesis contains individual and combined CAPTCHA-breakers for the darknet platforms
White House Market and ToRReZ Market.

### Usage

#### 1. Object-based CAPTCHA-breaker for WHM can be found in `01_WHM/`

- `image_extractor.py`: Extracting all images from one CAPTCHA in HTML
- `create_training_data.py`: Data preprocessing and preparation of training data
- `network.py`: Network architecture of CNN
- `cnn.py`: Training of network & storing of trained model
- `captcha_tester.py`: Measuring of test accuracy for test data
- `captcha_solver.py`: Solver returning results for provided CAPTCHA

#### 2. Text-based CAPTCHA-breaker for TRZ can be found in `02_TRZ/`

- `utils/`: CAPTCHA generator
- `image_splitter.py`: Data preprocessing and preparation of training data
- `helpers.py`: Helper functions for data preprocessing
- `network.py`: Network architecture of CNN
- `cnn.py`: Training of network & storing of trained model
- `captcha_tester.py`: Measuring of test accuracy for test data
- `captcha_solver.py`: Solver returning results for provided CAPTCHA

#### 3. Merged CAPTCHA-breaker for both WHM and TRZ can be found in `03_merged/`
- `network.py`: Generic network architecture of CNN
- `cnn.py`: Generic training of network & storing of trained model


### Data & trained models
The training data used and already trained models can be found here: https://git.sit.fraunhofer.de/york.yannikos/bachelor-wei 






