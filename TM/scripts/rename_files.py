import os

if __name__ == "__main__":
  print(os.listdir())
  list_of_files = os.listdir()
  count = 1

  for file in list_of_files:
    if(file != "rename_files.py"):
      new_filename = str(count) + ".png"
      os.rename(file, new_filename)
      count += 1
