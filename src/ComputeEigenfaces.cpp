/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

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



static void load_images(const string& dirname, vector<Mat>& images, vector<int>& labels) {
  fs::path dir(dirname);
  fs::directory_iterator it(dir), eod;
  BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod)) {
    cout << p << endl;
    if (is_regular_file(p)) {
      Mat image = imread(p.c_str(), 0);
      if (image.cols != 64 && image.cols != 64) {
	cout << p << endl;
	cout << image.size() << endl;
      }
      images.push_back(image);
      labels.push_back(0);
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

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 4) {
        cout << "usage: " << argv[0] << " <input_dir>> <output_dir> <num_eigenfaces>" << endl;
        exit(1);
    }


    string input_dir = string(argv[1]);
    string output_folder = string(argv[2]);
    int num_eigenfaces = atoi(argv[3]);

    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        load_images(input_dir, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening dir \"" << input_dir << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    int height = images[0].rows;
    int width = images[0].cols;

    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    cout << "Beginning training" << endl;
    model->train(images, labels);

    Mat eigenvalues = model->getMat("eigenvalues");
    Mat W = model->getMat("eigenvectors");
    Mat mean = model->getMat("mean");

    // Display or save the Eigenfaces:
    for (int i = 0; i < min(num_eigenfaces, W.cols); i++) {
      cout << i << endl;
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
	//        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        // Display or save:
	imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(grayscale));
	cout << format("%s/eigenface_%d.png", output_folder.c_str(), i) << endl;
    }

    return 0;
}
