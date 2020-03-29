import os
import glob
import csv
import re


def format_data(sub_folders):
    """
    Moves videos to target folder for model to train on.

    Parameter
    ---------
        video_path: path to collected video with json target text included.
    """
    # loop throught all sub-folders
    count = 0
    for videos_path in sub_folders:

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
            count += 1
            
    print("total data points: ", count)

if __name__ == "__main__":
    sub_folders = ["processed_data/downloaded/", "processed_data/goals/",
                    "processed_data/holiday/", "processed_data/school/",
                    "processed_data/summer/", "processed_data/vocab/"]

    format_data(sub_folders)