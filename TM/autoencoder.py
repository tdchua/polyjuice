import time
import os
import matplotlib as mpl
import numpy as np
from PIL import Image as img
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt


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



if __name__ == "__main__":

  os.system("clear")

  encoding                  = True
  training_A                = False
  training_A_dump           = False
  training_B                = False
  load_training_A           = False
  training_B_dump           = False
  already_trained_model     = False
  load_training_A           = False
  load_training_B           = False

  image_size                = 128

  if(encoding == True):
    print("Encoding Phase")
    path_to_data_a = "../data/tim/extract/size_change"
    data_a = []
    data_a = load_images_from_folder(path_to_data_a, data_a)


    path_to_data_b = "../data/harrypotter/extract/size_change"
    data_b = []
    data_b = load_images_from_folder(path_to_data_b, data_b)

    training_data = data_b
    print(len(training_data))


  if(training_A == True):

    input_img = Input(shape=(image_size,image_size,3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='enc_conv2d_1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='enc_maxpool_1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv2d_2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='enc_maxpool_2')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv2d_3')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='enc_maxpool_3')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_A_conv2d_1')(encoded)
    x = UpSampling2D((2, 2), name='dec_A_upsampl_1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_A_conv2d_2')(x)
    x = UpSampling2D((2, 2), name='dec_A_upsampl_2')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='dec_A_conv2d_3')(x)
    x = UpSampling2D((2, 2), name='dec_A_upsampl_3')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='dec_A_conv2d_4')(x)

    #MODEL Instantiation : https://blog.keras.io/building-autoencoders-in-keras.html

    autoencoder = Model(input_img, decoded)
    autoencoder = multi_gpu_model(Model)
    autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

    #Stats of the network https://laid.delanover.com/debugging-a-keras-neural-network/
    print(autoencoder.layers)
    print(autoencoder.summary())
    for layer in autoencoder.layers:
      print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))

    print("Training Phase")
    norm_training_data = [x/255 for x in training_data] #normalizing the data
    x_train = np.reshape(norm_training_data, (len(training_data), image_size, image_size, 3))
    autoencoder.fit(x_train,
                    x_train,
                    # epochs=1,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.25)

    if(training_A_dump == True):
      autoencoder.save_weights('weights_decoder_A.h5')

  if(training_B == True):

    input_img = Input(shape=(image_size,image_size,3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='enc_conv2d_1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='enc_maxpool_1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv2d_2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='enc_maxpool_2')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv2d_3')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='enc_maxpool_3')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_B_conv2d_1')(encoded)
    x = UpSampling2D((2, 2), name='dec_B_upsampl_1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_B_conv2d_2')(x)
    x = UpSampling2D((2, 2), name='dec_B_upsampl_2')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='dec_B_conv2d_3')(x)
    x = UpSampling2D((2, 2), name='dec_B_upsampl_3')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='dec_B_conv2d_4')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

    if(load_training_A == True):
      autoencoder.load_weights('weights_decoder_A.h5', by_name = True)

    print("Training Phase")
    norm_training_data = [x/255 for x in training_data] #normalizing the data
    x_train = np.reshape(norm_training_data, (len(training_data), image_size, image_size, 3))
    autoencoder.fit(x_train,
                    x_train,
                    # epochs=1,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.25)

    if(training_B_dump == True):
      autoencoder.save_weights('weights_decoder_B.h5')

  if(already_trained_model == True):

    input_img = Input(shape=(image_size,image_size,3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='enc_conv2d_1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='enc_maxpool_1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv2d_2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='enc_maxpool_2')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv2d_3')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='enc_maxpool_3')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_A_conv2d_1')(encoded)
    x = UpSampling2D((2, 2), name='dec_A_upsampl_1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_A_conv2d_2')(x)
    x = UpSampling2D((2, 2), name='dec_A_upsampl_2')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='dec_A_conv2d_3')(x)
    x = UpSampling2D((2, 2), name='dec_A_upsampl_3')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='dec_A_conv2d_4')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

    print(autoencoder.layers)
    print(autoencoder.summary())

    for layer in autoencoder.layers:
      print("Input shape: "+str(layer.input_shape)+". Output shape: "+str(layer.output_shape))


    if(load_training_A == True):
      autoencoder.load_weights('weights_decoder_a.h5', by_name=True)
    if(load_training_B == True):
      autoencoder.load_weights('weights_decoder_b.h5', by_name=True)

    digital_image = img.open("../data/tim/extract/size_change/6.png").convert("RGB")
    test_input = np.reshape(np.asarray(digital_image), (1, image_size, image_size, 3))
    my_output = autoencoder.predict(test_input / 255)
    my_output = np.reshape(my_output, (image_size, image_size, 3)) * 255
    # print(my_output[0])
    plt.imshow((my_output).astype(np.uint8))
    plt.show()
