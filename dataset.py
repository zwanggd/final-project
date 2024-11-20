# from SoccerNet.Downloader import SoccerNetDownloader
# mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/Users/kai/GSR/data")
# mySoccerNetDownloader.downloadDataTask(task="gamestate-2024",
#                                        split=["train"])

from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="/Users/kai/GSR/data/jersey_number")
mySNdl.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])

