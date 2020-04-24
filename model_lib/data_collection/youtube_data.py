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
    # create video directory if not exist
    if not os.path.exists(videos_dir):
        os.makedirs(transcripts_dir, exist_ok=True)

        playlist = Playlist(playlist_id)
        playlist.download_all(videos_dir)
    
    else:
        # skip downloading if videos already exists
        print("Videos from playlist with id {0} already downloaded, skipping...".format(playlist_id))


def download_transcripts_of_playlist(playlist_id, transcripts_dir):
    """Retrieves transcripts of individual youtube videos from playlist
    and writes to json files.
    params:
        playlist_id: youtube videos playlist id
        transcripts_dir: path to write transcripts json
    """
    # create transcripts directory if not exist
    if not os.path.exists(transcripts_dir):
        os.makedirs(transcripts_dir, exist_ok=True)

        # get video ids from playlist id
        video_ids = get_video_ids(playlist_id)

        # get transcripts
        transcripts, unretrieved_videos  = YouTubeTranscriptApi.get_transcripts(video_ids, languages=['en'], continue_after_error=True)
        count = 1
        for key, transcript in transcripts.items():
            # the json file where the output must be stored  
            file_path = os.path.join(transcripts_dir, str(count)+key+".json")
            print(file_path)
            out_file = open(file_path, "w")  
            # dump to json
            json.dump(transcript, out_file, indent = 3)     
            out_file.close()
            count+=1  

        print("unretrieved_videos: ", unretrieved_videos)

    else:
        # skip downloading transcripts if already exists
        print("playlist with id {0} transcripts already downloaded, skipping...".format(playlist_id))

def get_videos_and_transcripts(playlist_id, videos_dir, transcripts_dir):
    """Downloads entire videos and their corresponding transcripts
    in youtube playlists
    params:
        playlist_id: id of youtube playlist to download
        videos_dir: path to download videos to
        transcripts_dir: path to write transcripts json
    """
    # generate paths for individuals videos
    playlist_video_dir = os.path.join(videos_dir, playlist_id[0])
    playlist_video_transcript_dir = os.path.join(transcripts_dir, playlist_id[0])

    # download videos
    download_videos_of_playlist(playlist_id[1], playlist_video_dir)
    # download transcripts
    download_transcripts_of_playlist(playlist_id[1], playlist_video_transcript_dir)



if __name__ == "__main__":
    videos_dir = "downloads/videos/"
    transcripts_dir = "downloads/transcripts/"

    # playlists_ids = [("Signing about Schools", "PL-6MutNucAZI-WiheIHNjB_-Ho3UyJAew")]
    playlists_ids = [
                        ("Signing about Schools", "PL-6MutNucAZI-WiheIHNjB_-Ho3UyJAew"),
                        ("2020", "PL-6MutNucAZJXcRxcn1qWwq9Uer9e6XQH"),
                        ("Numbers in ASL", "PL-6MutNucAZL1mqZjOklkFgjaOtq-rWLI"),
                        ("Holidays", "PL-6MutNucAZLVzN3iqC30UMAYOjyg19vh"),
                        ("Family and Relationship ASL Vocabs", "PL-6MutNucAZIpABgLj2ZwC8zPlYXQENCm"),
                        ("Vocabulary", "PL-6MutNucAZK2pI2W8_MR_LNwSTR0GEIp"),
                        ("Summer Signs", "PL-6MutNucAZJlSTh4-f5-Q1ob-vsH82Bt"),
                        ("Describing your Home", "PL-6MutNucAZKniv9434D1qwimcTESRxht"),
                    ]

    for playlist_id in playlists_ids:
        get_videos_and_transcripts(playlist_id, videos_dir, transcripts_dir)