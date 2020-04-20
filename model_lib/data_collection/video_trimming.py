import os
import glob
import json
import csv

import moviepy
from moviepy.editor import VideoFileClip
import pandas as pd




def trim_videos(videos_path, target_data_path, processed_datapath):
    """Reads both videos and their corresponding transcription
    and trims them using the sentence duration.

    params:
        videos_path: path to downloaded videos
        target_data_path: path to corresponding transcription in json format
    """
    # read files
    videos = sorted(glob.glob(videos_path + "/*.mp4"))
    json_data = sorted(glob.glob(target_data_path + "/*.json"),
                         key = lambda x: int(x.split("\\")[-1].split(".")[0]))

    # check data is correct
    if len(videos) != len(json_data):
        print("number of videos and json items not equal")
        return
        
    # create folder to store processed clips 
    processed_datapath = os.path.join(processed_datapath, videos_path.split("\\")[1])
    if not os.path.exists(processed_datapath):
        os.makedirs(processed_datapath)

    # open csv file to store target data
    target_path = os.path.join(processed_datapath, videos_path.split("\\")[1]+".csv" )
    with open(target_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(videos)):
            # video clip
            my_clip = VideoFileClip(videos[i])

            # json data
            with open(json_data[i], 'r') as f:
                parsed_data = json.load(f)

            # save clip and corresponding target data
            for j, segment in enumerate(parsed_data):
                # generate clip file path
                file_name = json_data[i].split("\\")[-1].split(".")[0]+"_"+str(j)
                clip_path = os.path.join(processed_datapath, file_name+".mp4")

                # check if file exists
                if os.path.exists(clip_path):
                    print("clip already created, skipping...", clip_path)
                    continue
                
                # get duration of clip and check if clip is less than 5sec long
                duration = segment["duration"]
                if duration <= 5.0:
                    # generate clip
                    text = segment["text"]
                    start = segment["start"]
                    end = start + duration
                    try:
                        new_clip = my_clip.subclip(start, end)

                        # save clip
                        # print("saving clip {0} to path".format(i))
                        new_clip.write_videofile(clip_path)
                        print([file_name, text])

                        # save text
                        text = text.rstrip()
                        csv_writer.writerow([file_name, text])

                    except:
                        print("error occured, skipping...")
                        pass
                
            # close instance 
            my_clip.close()



if __name__ == "__main__":
    vid = ["videos\goals", "videos\holiday", "videos\school", "videos\summer", "videos\\vocab"]
    target = ["target_data\goals", "target_data\holiday",
                "target_data\school", "target_data\summer", "target_data\\vocab"]
    for i in range(len(vid)):
        videos_path = vid[i]
        target_data_json = target[i]
        processed_datapath = "processed_data"
        print("==========processing folder: ", videos_path, target_data_json)
        trim_videos(videos_path, target_data_json, processed_datapath)