# if downloader doesn't work, you can download office-31 dataset from
# https://people.eecs.berkeley.edu/~jhoffman/domainadapt/
# https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view
import gdown

if __name__ == "__main__":
    # TAKE ID FROM SHAREABLE LINK
    URL = "https://drive.google.com/uc?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE"
    # DESTINATION FILE ON YOUR DISK
    DESTINATION = "data/office-31/domain_adaptation_images.tar.gz"
    MD5_HASH = '1b536d114869a5a8aa4580b89e9758fb'
    gdown.cached_download(URL,
                          DESTINATION,
                          md5=MD5_HASH,
                          postprocess=gdown.extractall)
