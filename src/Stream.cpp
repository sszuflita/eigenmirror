
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

int top_half_factor, bottom_half_factor, width_factor;
const int max_factor = 200;
Rect face_rect;
Mat clean_frame;

static int find_face(CascadeClassifier& haar_cascade, Mat& gray_frame) {
  vector< Rect_<int> > faces;
  haar_cascade.detectMultiScale(gray_frame, faces);

  if (faces.size() == 0) {
    cout << "No faces found" << endl;
    return -1;
  }

  cout << faces.size() << " faces found." << endl;

  face_rect = faces[0];

  for (int i = 1; i < faces.size(); i++) {
    Rect face_i = faces[i];

    if (face_rect.area() < face_i.area()) {
      face_rect = face_i;
    }
  }
  return 1;
}

static void load_images(const string& dirname, vector<Mat>& images) {
  fs::path dir(dirname);
  fs::directory_iterator it(dir), eod;
  BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod)) {
    if (is_regular_file(p)) {
      cout << p << endl;
      Mat image = imread(p.c_str(), 0);
      images.push_back(image);
    }
  }
}

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

/**
 * @function on_trackbar
 * @brief Callback for trackbar
 */
void on_trackbar( int, void* ) {
  Rect face_rect_modified(face_rect);

  int new_width = face_rect.width * width_factor / 100;
  int x_center = face_rect.x + face_rect.width / 2;
  int new_x = x_center - new_width / 2;
  face_rect_modified.width = new_width;
  face_rect_modified.x = new_x;
  
  int y_center = face_rect.y + face_rect.height / 2;
  int new_y = y_center - (face_rect.height / 2) * top_half_factor / 100;
  int new_bottom = y_center + (face_rect.height / 2) * bottom_half_factor / 100;
  int new_height = new_bottom - new_y;
  face_rect_modified.height = new_height;
  face_rect_modified.y = new_y;
  
  Mat dirty_frame;

  clean_frame.copyTo(dirty_frame);

  rectangle(dirty_frame, face_rect_modified, CV_RGB(0, 255, 0), 1);
  imshow("RawFeed", dirty_frame);
}

static Mat load_mean(const string& mean_file) {
  fs::path p(mean_file);
  return imread(p.c_str(), 0);
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    cout << "usage: " << argv[0] << " <input_dir> <mean_file> <num_eigenfaces> <haar_file>" << endl;
    exit(1);
  }

  string input_dir = string(argv[1]);
  string mean_file = string(argv[2]);
  int num_eigenfaces = atoi(argv[3]);
  string haar_file = string(argv[4]);

  CascadeClassifier haar_cascade;
  haar_cascade.load(haar_file);

  cout << "Loading images" << endl;
  vector<Mat> images;

  try {
    load_images(input_dir, images);
  } catch (cv::Exception& e) {
    cerr << "Error opening dir \"" << input_dir << "\". Reason: " << e.msg << endl;
    exit(1);
  }
  cout << "Done loading images" << endl;

  int eigen_height = images[0].rows;
  int eigen_width = images[0].cols;
  int eigen_area = eigen_width * eigen_height;

  Mat mean = load_mean(mean_file);

  Mat W(eigen_area, num_eigenfaces, images[0].type());

  for (int i = 0; i < num_eigenfaces; i++) {
    Mat col = images[i].reshape(1, eigen_area);
    col.copyTo(W.col(i));
  }

  VideoCapture cap(0); // open the video camera no. 0

  if (!cap.isOpened()) {
    cout << "Cannot open the video cam" << endl;
    return -1;
  }

  double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
  double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

  namedWindow("RawFeed",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
  namedWindow("EigenFeed", CV_WINDOW_AUTOSIZE);
  namedWindow("FoundFace", CV_WINDOW_AUTOSIZE);

  Mat gray_frame;

  bool bSuccess = cap.read(clean_frame); // read a new frame from video

  if (!bSuccess) {
    cout << "Cannot read a frame from video stream" << endl;
  }

  cout << "Camera is working" << endl;

  width_factor = 100;
  top_half_factor = 100;
  bottom_half_factor = 100;

  createTrackbar( "Width", "RawFeed", &width_factor, max_factor, on_trackbar );
  createTrackbar( "Top", "RawFeed", &top_half_factor, max_factor, on_trackbar );
  createTrackbar( "Bottom", "RawFeed", &bottom_half_factor, max_factor, on_trackbar );

  cout << "Trackbars are working" << endl;

  cvtColor(clean_frame, gray_frame, CV_BGR2GRAY);

  cout << "Looking for dem faces" << endl;

  int face_found = find_face(haar_cascade, gray_frame);
  while (face_found == -1) {
      bool bSuccess = cap.read(clean_frame); // read a new frame from video

      if (!bSuccess) {
        cout << "Cannot read a frame from video stream" << endl;
      }

      cvtColor(clean_frame, gray_frame, CV_BGR2GRAY);

      imshow("RawFeed", clean_frame);
      face_found = find_face(haar_cascade, gray_frame);
  }

  Mat face = gray_frame(face_rect);
  Mat face_resized;
  cv::resize(face, face_resized, Size(eigen_width, eigen_height), 1.0, 1.0, INTER_CUBIC);

  face_resized.convertTo(face_resized, CV_32FC1);
  mean.convertTo(mean, CV_32FC1);
  W.convertTo(W, CV_32FC1);

  /*Mat projection = subspaceProject(W, mean, face_resized.reshape(1, 1));
  Mat reconstruction = subspaceReconstruct(W, mean, projection);

  reconstruction = norm_0_255(reconstruction.reshape(1, height));
  reconstruction.convertTo(reconstruction, CV_8UC1);

  mean.convertTo(mean, CV_8UC1);*/

  face_resized.convertTo(face_resized, CV_8UC1);

  imshow("RawFeed", clean_frame);
  imshow("FoundFace", face_resized);
//  imshow("EigenFeed", reconstruction.reshape(1, height));
  waitKey(0);
  return 0;
}
