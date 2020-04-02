import os
import glob
import csv
import re
import shutil


def format_data(sub_folders, dest_video_dir, dest_label_dir):
    """
    Moves videos to target folder for model to train on.

    Parameter
    ---------
        video_path: path to collected video with json target text included.
    """
    # create folder to store formated videos
    if not os.path.exists(dest_video_dir):
        os.makedirs(dest_video_dir)
        # os.makedirs(dest_label_dir)

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

        # create csv file to store target text
        with open(dest_label_dir, mode='a+', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # loop and copy videos to dataset folder for training
            for vid in videos:
                key = vid.split("\\")[-1].split(".")[0]
                text = data_dict[key]
                dest = os.path.join(dest_video_dir, str(count)+"."+vid.split("\\")[-1].split(".")[-1])
                # copy to destination
                print(count, text, "copying ",vid + "...")
                shutil.copy(vid, dest)
                # write to csv
                csv_writer.writerow([count, text])

                count += 1
            
    print("total data points: ", count)

if __name__ == "__main__":
    # sub_folders = ["processed_data/downloaded/", "processed_data/goals/",
    #                 "processed_data/holiday/", "processed_data/school/",
    #                 "processed_data/summer/", "processed_data/vocab/"]
    sub_folders = ["processed_data/goals/",
                    "processed_data/holiday/", "processed_data/school/",
                    "processed_data/summer/", "processed_data/vocab/"]

    dest_video_dir = "../dataset"
    dest_label_dir = "../dataset_labels.csv"
    format_data(sub_folders, dest_video_dir, dest_label_dir)