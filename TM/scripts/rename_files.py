#Created by Timothy Joshua Dy Chua
#A script used to rename files on a directory and to move it

import os

if __name__ == "__main__":
  print(os.listdir())
  list_of_files = os.listdir()  #This lists all of the files in a directory
  count = 1

  for file in list_of_files:    #Iterate through each of the file in the directory
    if(file != "rename_files.py"):
      new_filename = str(count) + ".png"  #New filename construction
      os.rename(file, new_filename)
      count += 1
