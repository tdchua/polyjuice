import time
import os
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
    if(image[-3:] == "png"):
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

  encoding                  = False
  training_A                = False
  training_A_dump           = False
  training_B                = False
  load_training_A           = False
  training_B_dump           = False
  already_trained_model     = True
  load_training_A           = False
  load_training_B           = True

  image_size                = 64

  if(encoding == True):
    print("Encoding Phase")
    path_to_data_a = "../data/tim/extract/size_change_64px"
    data_a = []
    data_a = load_images_from_folder(path_to_data_a, data_a)


    path_to_data_b = "../data/harrypotter/extract/size_change_64px"
    data_b = []
    data_b = load_images_from_folder(path_to_data_b, data_b)

    if(training_A):
      training_data = data_a
    if(training_B):
      training_data = data_b

    print(len(training_data))

  if(training_A == True):
    x = Input(shape=(64,64,3))
    my_encoder = Encoder()
    my_decoder_A = Decoder()
    optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

    auto_encoder = Model(x, my_decoder_A(my_encoder(x)))

    #MODEL Instantiation : https://blog.keras.io/building-autoencoders-in-keras.html
    auto_encoder.compile(optimizer=optimizer, loss='mean_absolute_error')

    #Stats of the network https://laid.delanover.com/debugging-a-keras-neural-network/
    print("Encoder#################")
    for layer in my_encoder.layers:
      print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))
    print("Decoder#################")
    for layer in my_decoder_A.layers:
      print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))
    print("Autoencoder##############")
    # for layer in auto_encoder.layers:
    #   print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))
    print(my_encoder.summary())
    print(my_decoder_A.summary())
    print(auto_encoder.summary())

    print("\n\n\nTraining Phase")
    norm_training_data = [x/255 for x in training_data] #normalizing the data
    x_train = np.reshape(norm_training_data, (len(training_data), image_size, image_size, 3))
    auto_encoder.fit(x_train,
                    x_train,
                    # epochs=1,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.25)

    if(training_A_dump == True):
      my_encoder.save_weights('weights_encoder.h5')
      my_decoder_A.save_weights('weights_decoder_A.h5')

  if(training_B == True):

    x = Input(shape=(64,64,3))
    my_encoder = Encoder()
    my_decoder_B = Decoder()
    optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

    auto_encoder = Model(x, my_decoder_B(my_encoder(x)))
    #MODEL Instantiation : https://blog.keras.io/building-autoencoders-in-keras.html
    auto_encoder.compile(optimizer=optimizer, loss='mean_absolute_error')

    if(load_training_A == True):
      my_encoder.load_weights('weights_encoder.h5')

    print("Training Phase")
    norm_training_data = [x/255 for x in training_data] #normalizing the data
    x_train = np.reshape(norm_training_data, (len(training_data), image_size, image_size, 3))
    auto_encoder.fit(x_train,
                    x_train,
                    # epochs=1,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.25)

    if(training_B_dump == True):
      my_encoder.save_weights('weights_encoder.h5')
      my_decoder_B.save_weights('weights_decoder_B.h5')


  if(already_trained_model == True):

    x = Input(shape=(64,64,3))
    my_encoder = Encoder()
    my_decoder_A = Decoder()
    my_decoder_B = Decoder()
    optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

    auto_encoder = Model(x, my_decoder_B(my_encoder(x)))
    #MODEL Instantiation : https://blog.keras.io/building-autoencoders-in-keras.html
    auto_encoder.compile(optimizer=optimizer, loss='mean_absolute_error')

    print(auto_encoder.layers)
    print(auto_encoder.summary())

    for layer in auto_encoder.layers:
      print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))


    if(load_training_A == True):
      my_encoder.load_weights('weights_encoder.h5')
      my_decoder_A.load_weights('weights_decoder_A.h5')
    if(load_training_B == True):
      my_encoder.load_weights('weights/weights_encoder.h5')
      my_decoder_B.load_weights('weights/weights_decoder_B.h5')

    image_number = '1'
    while(image_number != "stop"):
      digital_image = img.open("../data/tim/extract/size_change_64px/" + image_number + ".png").convert("RGB")
      test_input = np.reshape(np.asarray(digital_image), (1, image_size, image_size, 3))

      my_output = auto_encoder.predict(test_input / 255)
      my_output = np.reshape(my_output, (image_size, image_size, 3)) * 255
      # print(my_output[0])
      plt.imshow((my_output).astype(np.uint8))
      plt.show()
      image_number = input("Input image number: ")
