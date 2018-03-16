#!/usr/bin/env python3
"""
Scripts to check tubs

Usage:
    check_model.py [--tub=<tub1,tub2,..tubn>]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
"""

import PIL
import keras
import numpy as np
import donkeycar as dk
from docopt import docopt
import shutil
import os
from donkeycar.parts.datastore import TubGroup


def clean_slow_frames():
    args = docopt(__doc__)
    tubs = args["--tub"]
    tg = TubGroup(tubs)
    print(tg.df)
    tg.df[tg.df["user/throttle"] < 0.1]
    json_files_to_clean = tg.df[tg.df["user/throttle"] < 0.1]["cam/image_array"].map(
        lambda x: "\\".join(x.split("\\")[:-1]) + "\\" + "record_" +
                  x.replace("_cam-image_array_.jpg", "").rsplit("\\")[-1] + ".json")
    images_to_clean = tg.df[tg.df["user/throttle"] < 0.1]["cam/image_array"]
    for file in json_files_to_clean:
        f = "C:\\Users\\moritz\\Documents\\donkeycar\\d2\\data\\tmp\\" + file.split("\\")[-2] + file.split("\\")[-2]
        if not os.path.exists(f):
            os.mkdir(f)
        shutil.move(file, f)
    for file in images_to_clean:
        f = "C:\\Users\\moritz\\Documents\\donkeycar\\d2\\data\\tmp\\" + file.split("\\")[-2] + file.split("\\")[-2]
        if not os.path.exists(f):
            os.mkdir(f)
        shutil.move(file, f)

def estimate_frames():
    args = docopt(__doc__)
    tubs = args["--tub"]
    tg = TubGroup(tubs)
    #print(tg.df)
    model_path = "C:\\Users\\moritz\\Documents\\donkeycar\\d2\\mymodel_tmp2"
    model = keras.models.load_model(model_path)

    for index, frame in tg.df.iterrows():

        img = PIL.Image.open(frame["cam/image_array"])
        arr = np.array(img)
        n = np.uint8(arr)
        img_arr = n.reshape((1,) + n.shape)
        angle_binned, throttle = model.predict(np.uint8(img_arr))
        angle_unbinned = dk.utils.linear_unbin(angle_binned)
        print("angle_unbinned", angle_unbinned, "throttle", throttle)
        print("user/angle", frame["user/angle"], "user/throttle", frame["user/throttle"])

def estimate_image():
    img=PIL.Image.open("C:\\Users\\moritz\\Documents\\donkeycar\\d2\\data\\tub_77_18-02-10\\90_cam-image_array_.jpg")
    arr=np.array(img)

    #print(arr)
    #bin= img_to_binary(img)
    n=np.uint8(arr)
    img_arr = n.reshape((1,) + n.shape)
    model_path="C:\\Users\\moritz\\Documents\\donkeycar\\d2\\mymodel4"
    model=keras.models.load_model(model_path)
    angle_binned, throttle = model.predict(np.uint8(img_arr))
    angle_unbinned = dk.utils.linear_unbin(angle_binned)
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    print(angle_unbinned, throttle)

if __name__ == '__main__':
    #clean_slow_frames()
    #estimate_image()
    estimate_frames()