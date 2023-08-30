#!/usr/bin/env python3

import requests
import json
import base64
import os
from pathlib import Path


def get_captcha():
    r = requests.get("http://127.0.0.1:8000/captcha/api/flat")
    data = json.loads(r.text)

    key = data["key"]
    image = data["img"].split(",")[-1]
    working_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = working_dir + "/new_test_data/"

    print(test_data_dir)
    Path(test_data_dir).mkdir(parents=True, exist_ok=True)
    with open(test_data_dir + key.lower() + ".png", "wb") as img:
        img.write(base64.b64decode(image))


for i in range(1000):
    get_captcha()
    print(i)

