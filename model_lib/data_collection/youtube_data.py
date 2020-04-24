import json
import os

from youtube_transcript_api import YouTubeTranscriptApi
from pyyoutube import Api
from pytube import Playlist

# create api
api = Api(api_key='AIzaSyAUdxEy_nslLj9mTdOVKj5diUvs9OKtQsU')


def get_video_ids(playlist_id):
    """Retrieves individual video id from a playlist
    params:
        playlist_id: youtube playlist id
    """
    video_ids = []
    playlist_items  = api.get_playlist_items(playlist_id=playlist_id, count=None)
    for item in playlist_items.items:
        video_ids.append(item.snippet.resourceId.videoId)

    return video_ids


def download_videos_of_playlist(playlist_id, videos_dir):
    """Downloads entire videos in youtube video playlists
    params:
        playlist_id: id of youtube playlist to download
        videos_dir: path to download videos to
    """
    playlist = Playlist(playlist_id)
    playlist.download_all(videos_dir)


def download_transcripts_of_playlist(playlist_id, transcripts_dir):
    """Retrieves transcripts of individual youtube videos from playlist
    and writes to json files.
    params:
        playlist_id: youtube videos playlist id
        transcripts_dir: path to write transcripts json
    """
    # get video ids from playlist id
    video_ids = get_video_ids(playlist_id)

    # get transcripts
    transcripts, unretrieved_videos  = YouTubeTranscriptApi.get_transcripts(video_ids, languages=['en'], continue_after_error=True)

    for key, transcript in transcripts.items():
        # the json file where the output must be stored  
        file_path = os.path.join(transcripts_dir, key+".json")
        print(file_path)
        out_file = open(file_path, "w")  
        # dump to json
        json.dump(transcript, out_file, indent = 3)     
        out_file.close()  

    print("unretrieved_videos: ", unretrieved_videos)

def get_videos_and_transcripts(playlist_id, videos_dir, transcripts_dir):
    """Downloads entire videos and their corresponding transcripts
    in youtube playlists
    params:
        playlist_id: id of youtube playlist to download
        videos_dir: path to download videos to
        transcripts_dir: path to write transcripts json
    """
    # download videos
    download_videos_of_playlist(playlist_id, videos_dir)
    # download transcripts
    download_transcripts_of_playlist(playlist_id, transcripts_dir)



if __name__ == "__main__":
    videos_dir = "downloads/videos/"
    transcripts_dir = "downloads/transcripts/"

    playlists_ids = ["PL-6MutNucAZI-WiheIHNjB_-Ho3UyJAew"]
    for playlist_id in playlists_ids:
        get_videos_and_transcripts(playlist_id, videos_dir, transcripts_dir)