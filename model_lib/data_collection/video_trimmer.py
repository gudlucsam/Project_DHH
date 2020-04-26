import os
import glob
import json
import csv
import re

import moviepy
from moviepy.editor import VideoFileClip
import pandas as pd




def trim_videos(downloads_path, dataset_path):
    """Reads both videos and their corresponding transcription
    and trims them using the sentence duration in their transcripts.

    params:
        downloadeds_path: path to downloaded videos, and transcripts
        dataset_path: path to corresponding transcription in json format
    """

    # skip processing if folder already exists
    if not os.path.exists(dataset_path):
        # print("downloaded videos already processed to {0} folder, exiting...".format(dataset_path))
        # return

        # create folder to store processed clips as dataset
        os.makedirs(dataset_path)

    # base videos and transcripts paths
    videos_path = os.path.join(downloads_path, "videos/")
    transcripts_path = os.path.join(downloads_path, "transcripts/")

    # open csv file to store procesed target sentences
    target_sentences = os.path.join(dataset_path, "dataset_labels.csv")
    with open(target_sentences, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # traverse all sub folders in order
        data_point = 1
        for cat in os.listdir(transcripts_path):
            # sub folder dir path
            video_catdir = os.path.join(videos_path, cat)
            transcripts_catdir = os.path.join(transcripts_path, cat)

            # read files in order
            videos = sorted(glob.glob(video_catdir + "/*.mp4"))
            transcripts = sorted(glob.glob(transcripts_catdir + "/*.json"),
                                key = lambda x: int(x.split("\\")[-1].split("_")[0]))

            # check data is correct
            if len(videos) != len(transcripts):
                print("number of videos and json items not equal in {0} category, skipping...".format(cat))
                continue


            for i in range(len(videos)):
                # video clip
                my_clip = VideoFileClip(videos[i])

                # read transcription json data
                with open(transcripts[i], 'r') as f:
                    parsed_data = json.load(f)

                # save clip and corresponding target data
                for j, segment in enumerate(parsed_data):
                    # generate clip file path
                    file_name = str(data_point)
                    clip_path = os.path.join(dataset_path, file_name+".mp4")

                    # check if file exists
                    if os.path.exists(clip_path):
                        print("clip already created, skipping...", clip_path)
                        data_point+=1
                        continue
                    
                    # get duration of clip and check if clip is less than 5sec long
                    duration = segment["duration"]
                    if duration <= 7.0:
                        # generate clip
                        text = segment["text"]
                        start = segment["start"]
                        end = start + duration
                        try:
                            new_clip = my_clip.subclip(start, end)

                            # save clip
                            new_clip.write_videofile(clip_path)
                            print([file_name, text])

                            # save text
                            text = text.rstrip()
                            text = re.sub('[:,.)(?!;*"-]', "", text.lower())
                            csv_writer.writerow([file_name, text])
                            # move to next data point
                            data_point+=1 

                        except:
                            print("error occured, skipping...")
                    
                # close instance 
                my_clip.close()



if __name__ == "__main__":
    downloads_path = "downloads/"
    dataset_path = "dataset/"
    trim_videos(downloads_path, dataset_path)