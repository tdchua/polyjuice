#https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
#A script used to reshape images to the desired size for inputting it into the model.

import cv2
import os


def vid_to_frames(vid_file):
  vidcap = cv2.VideoCapture(vid_file)
  success,image = vidcap.read()
  count = 0

  while success:
    cv2.imwrite("sliced_frames/frame%d.jpg" % count, image)
    success, image = vidcap.read()
    print("Read a new frame: ", success)
    count += 1

if __name__ == "__main__":

  vid_to_frames("target.mov")
