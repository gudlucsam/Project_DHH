from pytube import Playlist


def download(playlist_id):
    """Downloads entire videos in youtube video playlists

    params:
        playlist_id: youtube video playlist to download
    """

    playlist = Playlist(playlist_id)
    playlist.download_all()



if __name__ == "__main__":
    playlist_id = "https://www.youtube.com/playlist?list=PL-6MutNucAZJlSTh4-f5-Q1ob-vsH82Bt"
    download(playlist_id)