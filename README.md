# eigenmirror

Software designed to perform face detection using Kirby and Sirovich's (1987) eigenface approach. The initial goal of this project is to build eigenface representations of camera input in real time.

With Haar detection, the processing time increases quite a bit. This may or may not be acceptable.

At this point, the eigenfaces seem to be inadequate to reconstruct the image. Explore new facial database.

Potential interface:

a wall of eigenfaces whose intensity corresponds to their weight

DONE:
a) train based on face database
b) compute eigen representation of a face
c) feed input from camera
d) benchmark b), determine feasibility of real time computation
e) include haar face detection

TODO:
a) experiment with larger face database / preprocessing of faces
b) experiment with display
c) experiment with face sliders