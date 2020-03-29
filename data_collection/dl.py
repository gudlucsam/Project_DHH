from pytube import Playlist


def download():
    playlist = Playlist("https://www.youtube.com/playlist?list=PL-6MutNucAZJlSTh4-f5-Q1ob-vsH82Bt")
    playlist.download_all()



if __name__ == "__main__":
    download()