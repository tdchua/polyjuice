# tim_potter_faceswap
This is an application made to swap my face with Daniel Radcliffe's(Harry Potter Actor).
This project was made to complete the requirement for the internship of **Thinking Machines**!
There's actually a detailed specification for this project labeled Documentation.pdf

## Files
* TM - Thinking Machines application folder
    * PixelShuffler.py - This model was used as a stage in the CNN Architecture pipeline. I edited the code I got from this link due to misusage of the [library](https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981)
    * faceswap.py - The machine learning model
    * list_of_faces - This contains the list of images with faces recognized by the another [module](https://github.com/ageitgey/face_recognition) I used.
    * Scripts
      * rename_files.py - A script that is used just to rename the photos to a number in ascending order.
      * resize_images.py - Just to change the photo sizes into 64x64 as what the model requires.
      * slice_video.py - Slices a video into frames. The model processes them frame by frame not as a complete video.
* Documentation.pdf - The detailed specification for this project.
