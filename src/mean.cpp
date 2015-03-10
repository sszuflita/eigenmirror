
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
    cout << p << endl;
    if (is_regular_file(p)) {
      Mat image = imread(p.c_str(), 0);
      images.push_back(image);
    }
  }
}

static void compute_mean(vector<Mat>& images, Mat& mean) {
  for (int i = 0; i < images.size(); i++) {
    cv::accumulate(images[i], mean);
  }
  mean = mean / images.size();
  mean.convertTo(mean, CV_8U);
}


int main(int argc, char* argv[]) {
  if (argc != 3) {
    cout << "usage: " << argv[0] << " <input_dir> <output_file>" << endl;
    exit(1);
  }

  string input_dir = string(argv[1]);
  string output_file = string(argv[2]);

  cout << "Loading images" << endl;
  vector<Mat> images;

  try {

    load_images(input_dir, images);
  } catch (cv::Exception& e) {
    cerr << "Error opening dir \"" << input_dir << "\". Reason: " << e.msg << endl;
    exit(1);
  }

  cout << "Done loading images" << endl;

  int height = images[0].rows;
  int width = images[0].cols;

  Mat mean(height, width, CV_32F);

  compute_mean(images, mean);

  namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

  imwrite(output_file, mean);

  return 0;
}
