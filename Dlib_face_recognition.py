# Dlib is downloaded from http://dlib.net/ and the following file is from face_recognition.py. The original file can be found at 
# ./dlib-19.24/python_examples/face_recognition.py. The file has been changed to remove/add comments and made changes/additions to the code. 

import sys
import os
import dlib
import glob

if len(sys.argv) != 1:  #Changed to 1 from 4. Can be changed back if you want to check.
    print(
        "Call this program like this:\n"
        "    ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
        "    You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

# Predictor_path and face_rec_model_path paths is the downloaded files from line 11-12. It has then been extracted to .dat from .bz2. 
predictor_path = "C:\\Users\\krist\\Downloads\\shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "C:\\Users\\krist\\Downloads\\dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "C:\\Users\\krist\\OneDrive\\Biometric systems\\Subjects\\100\\AM"

# Load all the models we need: a detector to find the faces, a shape predictor to find face landmarks so we can precisely localize the 
# face, and finally the face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

# Processes all the images. Changed from .jpeg to .png. 
for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the second argument indicates that we should upsample the 
    # image 1 time. This will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # This part processes each face we found.
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.
        # Data from this is used to see how the subjects position and other anatomical and biological factors (like beards, and glasses)
        # could influence results. 
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)

        # Computes the 128D vector that describes the face in img identified by shape.
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        print(face_descriptor)

        # Stores the face descriptor in a text file in the same folder as the picture to be used in DET script. 
        descriptor_file = os.path.splitext(f)[0] + "_descriptor.txt"
        with open(descriptor_file, "w") as f:
            for value in face_descriptor:
                f.write(str(value) + "\n")

    #dlib.hit_enter_to_continue()  # Is removed to automate the process, but was used at first to check the program and pictures.  