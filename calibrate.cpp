#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

vector< vector< Point3d > > object_points;
vector< vector< Point2f > > imagePoints1;
vector< Point2f > corners1;
vector< vector< Point2d > > img_points;

Mat img1, gray1;

//ディレクトリの中にあるpngとjpg画像を読み込んで、チェッカーボードを探す
void load_directory(int board_width, int board_height, float square_size, string directory) {
    Size board_size = Size(board_width, board_height);
    int board_n = board_width * board_height;

    vector<String> img_jpg, img_png;
    glob(directory + "*.jpg", img_jpg, false);
    glob(directory + "*.png", img_png, false);
    vector<String> imgs;
    size_t count_j = img_jpg.size();
    size_t count_p = img_png.size();

    for (size_t i = 0; i < count_j; i++)
        imgs.push_back(img_jpg.at(i));
    for (size_t i = 0; i < count_p; i++)
        imgs.push_back(img_png.at(i));


    for (int i = 0; i < imgs.size(); i++) {
        Mat img1 = imread(imgs.at(i), IMREAD_COLOR);
        if (img1.empty()) continue;
        cvtColor(img1, gray1, COLOR_BGR2GRAY);

        bool found1 = false;
        found1 = findChessboardCorners(img1, board_size, corners1,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);

        if (found1)
        {
            cornerSubPix(gray1, corners1, Size(5, 5), Size(-1, -1),
                TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
        }

        vector<Point3d> obj;
        for (int i = 0; i < board_height; ++i)
            for (int j = 0; j < board_width; ++j)
                obj.push_back(Point3d(double((float)j * square_size), double((float)i * square_size), 0));

        if (found1) {
            cout << i << " Found corners!" << endl;
            imagePoints1.push_back(corners1);
            object_points.push_back(obj);
        }
    }
    for (int i = 0; i < imagePoints1.size(); i++) {
        vector< Point2d > v1;
        for (int j = 0; j < imagePoints1[i].size(); j++) {
            v1.push_back(Point2d((double)imagePoints1[i][j].x, (double)imagePoints1[i][j].y));
        }
        img_points.push_back(v1);
    }
}

int main(int argc, char* argv[])
{
  //"calibrate.exe [チェスボードの交点の数(横)] [チェスボードの交点の数(縦)] [1マスの長さ] [キャリブ用画像が入ったディレクトリ]"
  int board_width = atoi(argv[1]), board_height = atoi(argv[2]);
  float square_size = atoi(argv[3]);
  string input_directory = argv[4];

  load_directory(board_width, board_height, square_size, input_directory);
  printf("Starting Calibration\n");

  //中心射影
  Mat K_pers, D_pers, r_pers, t_pers;
  double rms_pers = cv::calibrateCamera(object_points, img_points, img1.size(), K_pers, D_pers, r_pers, t_pers);

  //Fisheyeモデル
  Mat K_fish, D_fish, r_fish, t_fish;
  TermCriteria criteria_fish(TermCriteria::COUNT + TermCriteria::EPS, 200, 1e-9);
  int flags_fish = fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_FIX_SKEW;
  double rms_fish = fisheye::calibrate(object_points, img_points, img1.size(), K_fish, D_fish, r_fish, t_fish, flags_fish, criteria_fish);

  //Omnidirectionalモデル
  Mat K_om, D_om, r_om, t_om, xi;
  TermCriteria criteria_om(TermCriteria::COUNT + TermCriteria::EPS, 200, 1e-9);
  int flags_om = omnidir::CALIB_FIX_SKEW;
  double rms_om = omnidir::calibrate(object_points, img_points, img1.size(), K_om, xi, D_om, r_om, t_om, flags_om, criteria_om);
	
  cout << "RMS Perspective: " << rms_pers << endl;
  cout << "RMS Fisheye: " << rms_fish << endl;
  cout << "RMS Omnidirectional: " << rms_om << endl;
  return 0;
}
