#Created by Timothy Joshua Dy Chua
#This is my first ever Machine Learning project outside of school

import time
import os
import psutil
import face_recognition
import matplotlib as mpl
import numpy as np
from PixelShuffler import PixelShuffler
from PIL import Image as img
from keras import layers
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Flatten, Reshape


#https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
def load_images_from_folder(folder:str, data:list):

  start_encode = time.perf_counter()

  images = []
  list_of_files = os.listdir(folder)

  print("Folder: " + folder)

  for image in list_of_files:
    if(image[-3:] == "png" or image[-4:] == "jpeg"):
      digital_image = img.open(folder+'/'+image).convert("RGB")
      matrix_convert = np.asarray(digital_image)
      data.append(matrix_convert)

  end_encode = time.perf_counter()
  diff_encode = end_encode - start_encode
  print("Total Duration(s): ", diff_encode)

  return data

#https://medium.com/gradientcrescent/deepfaking-nicolas-cage-into-the-mcu-using-autoencoders-an-implementation-in-keras-and-tensorflow-ab47792a042f
#------>
def Encoder():
  input_ = Input(shape=(64, 64, 3))
  x = input_
  x = Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
  x = LeakyReLU(0.1)(x)
  x = Conv2D(256, kernel_size=5, strides=2, padding="same")(x)
  x = LeakyReLU(0.1)(x)
  x = Conv2D(512, kernel_size=5, strides=2, padding="same")(x)
  x = LeakyReLU(0.1)(x)
  x = Conv2D(1024, kernel_size=5, strides=2, padding="same")(x)
  x = LeakyReLU(0.1)(x)
  ENCODER_DIM = 1024
  x = Dense(ENCODER_DIM)(Flatten()(x))
  x = Dense(4 * 4 * 1024)(x)
  #Passed flattened X input into 2 dense layers, 1024 and 1024*4*4
  x = Reshape((4, 4, 1024))(x)
  #Reshapes X into 4,4,1024
  x = Conv2D(512 * 4, kernel_size=3, padding="same")(x)
  x = LeakyReLU(0.1)(x)
  x = PixelShuffler()(x)
  return Model(input_, x)

def Decoder():
  input_ = Input(shape=(8, 8, 512))
  x = input_
  x = Conv2D(256 * 4, kernel_size=3, padding="same")(x)
  x = LeakyReLU(0.1)(x)
  x = PixelShuffler()(x)
  x = Conv2D(128 * 4, kernel_size=3, padding="same")(x)
  x = LeakyReLU(0.1)(x)
  x = PixelShuffler()(x)
  x = Conv2D(64 * 4, kernel_size=3, padding="same")(x)
  x = LeakyReLU(0.1)(x)
  x = PixelShuffler()(x)
  x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(x)
  return Model(input_, x)
#<---------


if __name__ == "__main__":

  os.system("clear")

  #Runtime parameters. This is my own "jupyter" Although, I would port the code soon to a notebook.
  training_A                = False
  training_A_dump           = False
  training_B                = False
  load_training_A           = False
  training_B_dump           = False
  already_trained_model     = True
  load_training_A           = False
  load_training_B           = True

  image_size                = 64

  #Trains the Shared Encoder + Decoder A
  if(training_A == True):

    #This part loads the training data
    print("Encoding Phase")
    path_to_data_a = "../data/without_glasses/tim/extract/size_change_64px"
    data_a = []
    data_a = load_images_from_folder(path_to_data_a, data_a)

    #Prepares and instantiates the autoencoder model
    x = Input(shape=(64,64,3))
    my_encoder = Encoder()
    my_decoder_A = Decoder()
    auto_encoder = Model(x, my_decoder_A(my_encoder(x)))

    #Sets the paramters for the optimizer and the loss function
    optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
    auto_encoder.compile(optimizer=optimizer, loss='mean_absolute_error')

    #Stats of the network https://laid.delanover.com/debugging-a-keras-neural-network/
    print("Encoder#################")
    for layer in my_encoder.layers:
      print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))
    print("Decoder#################")
    for layer in my_decoder_A.layers:
      print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))
    print("Autoencoder##############")

    print(my_encoder.summary())
    print(my_decoder_A.summary())
    print(auto_encoder.summary())

    print("\n\n\nTraining Phase")
    norm_training_data = [x/255 for x in data_a] #normalize the data so it will converge faster
    x_train = np.reshape(norm_training_data, (len(data_a), image_size, image_size, 3))
    auto_encoder.fit( x_train,                    #Set the other training parameters and begin
                      x_train,
                      # epochs=1,
                      epochs=100,
                      batch_size=32,
                      shuffle=True,
                      validation_split=0.25 )

    #Dumping the weights
    if(training_A_dump == True):
      my_encoder.save_weights('weights/7564ef9/weights_encoder.h5')
      my_decoder_A.save_weights('weights/7564ef9/weights_decoder_A.h5')



  #Trains the Shared Encoder + Decoder B
  if(training_B == True):
    #This part loads the training data
    print("Encoding Phase")
    path_to_data_b = "../data/without_glasses/harrypotter/extract/size_change_64px"
    data_b = []
    data_b = load_images_from_folder(path_to_data_b, data_b)

    #Prepares and instantiates the autoencoder model
    x = Input(shape=(64,64,3))
    my_encoder = Encoder()
    my_decoder_B = Decoder()
    auto_encoder = Model(x, my_decoder_B(my_encoder(x)))

    #Sets the paramters for the optimizer and the loss function
    optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
    auto_encoder.compile(optimizer=optimizer, loss='mean_absolute_error')

    #Since this is a shared encoder, we have to load the weights trained in the first stage with decoder A
    if(load_training_A == True):
      my_encoder.load_weights('weights/7564ef9/weights_encoder.h5')

    print("Training Phase")
    norm_training_data = [x/255 for x in data_b] #normalize the data so it will converge faster
    x_train = np.reshape(norm_training_data, (len(data_b), image_size, image_size, 3))
    auto_encoder.fit( x_train,                    #Set the other training parameters and begin
                      x_train,
                      # epochs=1,
                      epochs=100,
                      batch_size=32,
                      shuffle=True,
                      validation_split=0.25 )

    #Dumping the weights
    if(training_B_dump == True):
      my_encoder.save_weights('weights_encoder.h5')
      my_decoder_B.save_weights('weights_decoder_B.h5')

  #Assumes that this model has been trained before, and weights can be loaded during execution
  if(already_trained_model == True):

    #Prepares and instantiates the autoencoder model
    x = Input(shape=(64,64,3))
    my_encoder = Encoder()
    my_decoder_A = Decoder()
    my_decoder_B = Decoder()
    #This uses the shared encoder + the decoder B to put Harry's face on mine.
    auto_encoder = Model(x, my_decoder_B(my_encoder(x)))

    #Sets the paramters for the optimizer and the loss function
    optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
    auto_encoder.compile(optimizer=optimizer, loss='mean_absolute_error')

    #For printing the model stats
    print(auto_encoder.layers)
    print(auto_encoder.summary())

    for layer in auto_encoder.layers:
      print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))

    #To load the weights for putting my face on Harry's face
    if(load_training_A == True):
      my_encoder.load_weights('weights_encoder.h5')
      my_decoder_A.load_weights('weights_decoder_A.h5')

    #To load the weights for putting Harry's face on mine
    if(load_training_B == True):
      my_encoder.load_weights('weights_encoder.h5')
      my_decoder_B.load_weights('weights_decoder_B.h5')


    #Let's get to frame rendering now :D
    #This opens the list_of_faces file where the filepath and face coordinates generated by the face_recognition module are listed.
    faces_file = open("list_of_faces")
    list_of_faces = []
    # # ------------>
    # for face_coordinate in faces_file:
    #
    #   #Cropping Section
    #   face_path, t,r,b,l = face_coordinate.split(',') #t:top, r:right, b:botom, l:left
    #   print(face_path, t,r,b,l)
    #   my_face = img.open(face_path)
    #   t,r,b,l = map(int,[t,r,b,l])
    #
    #
    #   #We adjust the cropped image to include the shape of my face and my hair
    #   # l = l - 50
    #   # t = t - 300
    #   # r = r + 50
    #   # b = b + 150
    #
    #   #To calculate the size of the cropped image, so that reconstruction will reshape the swapped faces to original size.
    #   orig_width = r - l
    #   orig_length = b - t
    #
    #   my_face_cropped = my_face.crop((l,t,r,b)) #Crops the image of my face from the image
    #   # my_face_cropped.show()
    #   # time.sleep(5)
    #
    #   #Resize Section
    #   my_small_face = my_face_cropped.resize((64,64))
    #   my_small_face_copy = my_small_face.copy()
    #   my_small_face = np.reshape(np.asarray(my_small_face), (1, image_size, image_size, 3)) #Changes it into 64,64,3
    #   # my_small_face_copy.show()
    #   # time.sleep(5)
    #
    #   #Normalizing the Photo
    #   my_small_face_norm = my_small_face / 255
    #
    #   #Generating the Harry/Tim Hybrid
    #   swap_to_harry = auto_encoder.predict(my_small_face_norm)
    #   swap_to_harry = np.reshape(swap_to_harry, (image_size, image_size, 3)) * 255
    #   swap_to_harry = swap_to_harry.astype(np.uint8)
    #   swap_harry_img = img.fromarray(swap_to_harry, 'RGB')
    #   # swap_harry_img.show()
    #   # time.sleep(5)
    #
    #   #Reshaping the output image
    #   my_harry_face = swap_harry_img.resize((orig_width,orig_length))
    #
    #   #Pasting the new face into the old picture
    #   my_face_copy = my_face.copy()
    #   my_face_copy.paste(my_harry_face, (l,t))
    #   face_path_split = face_path.split('/')
    #
    #   my_face_copy.save(face_path_split[0] + '/' + face_path_split[1] + '/' + "faceswap_frames" + '/' + face_path_split[3])
    #   # my_face_copy.show()
    #   # time.sleep(5)
    # #<------------


    image_number = '1'
    while(image_number != "stop"):
      digital_image = img.open("../data/without_glasses/tim/extract/size_change_64px/" + image_number + ".jpeg").convert("RGB")
      test_input = np.reshape(np.asarray(digital_image), (1, image_size, image_size, 3))

      my_output = auto_encoder.predict(test_input / 255)
      my_output = np.reshape(my_output, (image_size, image_size, 3)) * 255
      # print(my_output[0])
      plt.imshow((my_output).astype(np.uint8))
      plt.show()
      image_number = input("Input image number: ")
