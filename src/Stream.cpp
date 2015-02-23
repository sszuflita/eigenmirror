
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp> 

using namespace cv;
using namespace std;
namespace fs = boost::filesystem; 

static void load_images(const string& dirname, vector<Mat>& images) {
  fs::path dir(dirname);
  fs::directory_iterator it(dir), eod;
  BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod)) {
    if (is_regular_file(p)) {
      Mat image = imread(p.c_str(), 0);
      images.push_back(image);
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    cout << "usage: " << argv[0] << " <input_dir> <num_eigenfaces>" << endl;
    exit(1);
  }

  string input_dir = string(argv[1]);
  int num_eigenfaces = atoi(argv[2]);

    cout << "Loading images" << endl;
  vector<Mat> images;

  try {

    load_images(input_dir, images);
  } catch (cv::Exception& e) {
    cerr << "Error opening dir \"" << input_dir << "\". Reason: " << e.msg << endl;
    // nothing more we can do
    exit(1);
  }
  cout << "Done loading images" << endl;
    int height = images[0].rows;
    int width = images[0].cols;

    Mat W(num_eigenfaces, width * height, DataType<float>::type);
    for (int i = 0; i < num_eigenfaces; i++) {
      W.row(i) = images[i];
    }

  VideoCapture cap(0); // open the video camera no. 0

  if (!cap.isOpened())  // if not success, exit program
    {
      cout << "Cannot open the video cam" << endl;
      return -1;
    }

  double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
  double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

  cout << "Frame size : " << dWidth << " x " << dHeight << endl;

  namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

  while (1)
    {
      Mat frame;

      bool bSuccess = cap.read(frame); // read a new frame from video

      if (!bSuccess) //if not success, break loop
        {
	  cout << "Cannot read a frame from video stream" << endl;
	  break;
        }

      imshow("MyVideo", frame); //show the frame in "MyVideo" window

      if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
	{
	  cout << "esc key is pressed by user" << endl;
	  break; 
	}
    }
  return 0;

}
