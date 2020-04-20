import json
import os

from youtube_transcript_api import YouTubeTranscriptApi
from pyyoutube import Api



# create api
api = Api(api_key='AIzaSyAUdxEy_nslLj9mTdOVKj5diUvs9OKtQsU')

def get_video_ids(playlist_id):
    """Retrieves individual video id from a playlist

    params:
        playlist_id: youtube playlist id
    """

    video_ids = []
    playlist_item_by_playlist  = api.get_playlist_items(playlist_id=playlist_id, count=None)
    for item in playlist_item_by_playlist.items:
        video_ids.append(item.snippet.resourceId.videoId)

    return video_ids

def get_transcripts_data_as_json(playlist_id, folder_name):
    """Retrieves transcripts of individual youtube videos from playlist
    and writes to json files.

    params:
        playlist_id: youtube videos playlist id
        folder_name: path to write transcripts json
    """
    transcripts, unretrieved_videos  = YouTubeTranscriptApi.get_transcripts(video_ids, languages=['en'], continue_after_error=True)

    for key, transcript in transcripts.items():
        # the json file where the output must be stored  
        file_path = os.path.join(folder_name, key+".json")
        print(file_path)
        out_file = open(file_path, "w")  
        # dump to json
        json.dump(transcript, out_file, indent = 4)     
        out_file.close()  

    print(unretrieved_videos)


if __name__ == "__main__":
    playlist_id = "PL-6MutNucAZJXcRxcn1qWwq9Uer9e6XQH"
    json_folder = "2020_goals"
    video_ids = get_video_ids(playlist_id)
    get_transcripts_data_as_json(video_ids, json_folder)