import os
import glob
import csv
import re


def format_data(videos_path):
    """
    Moves videos to target folder for model to train on.

    Parameter
    ---------
        video_path: path to collected video with json target text included.
    """
    videos = sorted(glob.glob(videos_path + "/*.mp4"))
    csv_text = sorted(glob.glob(videos_path + "/*.csv"))

    # read in csv data
    data_dict = {}
    with open(csv_text[0], newline='') as f:
        reader = list(csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))
        for ln in reader:
            data_dict[ln[0]] = re.sub('[:,.)(?!;*"-]', "", ln[1].lower()) 

    for vid in videos:
        key = vid.split("\\")[-1].split(".")[0]
        print(vid, data_dict[key])


if __name__ == "__main__":
    videos_path = "processed_data/goals/"
    format_data(videos_path)