import os
from PIL import Image as img

if __name__ == "__main__":
  list_of_files = os.listdir()

  for file in list_of_files:
    if(file[-3:] == "png"):
      my_image = img.open(file)
      my_resized_image = my_image.resize((28,28))
      my_resized_image.save("size_change/" + file)
      my_image.close()
