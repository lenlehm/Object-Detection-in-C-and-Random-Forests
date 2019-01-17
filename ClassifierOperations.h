//
// Created by Rajdeep Surolia on 2019-01-04.
//

#ifndef TDCV_CLASSIFIEROPERATIONS_H
#define TDCV_CLASSIFIEROPERATIONS_H

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/dir.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "BoundingBoxOperations.h"
#include "opencv2/ml.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;


vector<Ptr<DTrees>> create_forest(int, int, int, int);
vector<Ptr<DTrees>> train_forest(vector<Ptr<DTrees>>, vector<Mat>, vector<int>, float [], int);
float predict_forest(vector<Ptr<DTrees>>, vector<Mat>, vector<int>, int, int, vector<int> *, vector<float> *);

//GLOBAL VARIABLES
int RESIZED_IMG = 160;
int BLOCK_SIZE = 40;
int BLOCK_STRIDE = 20;
int CELL_SIZE = 20;
int NUM_BINS = 9;


vector<Ptr<DTrees>> create_forest(int num_trees, int depth, int batch_size, int min_samp_count_perc){
    int i;
    vector<Ptr<DTrees>> trees;

    for(i=0; i<num_trees;i++){
        Ptr<DTrees> model = DTrees::create();
        model->setCVFolds(1);
        model->setMaxCategories(10);
        model->setMaxDepth(depth);
        model->setMinSampleCount(int(batch_size / min_samp_count_perc));

        trees.push_back(model);
    }

    return trees;
}

vector<Ptr<DTrees>> train_forest(vector<Ptr<DTrees>> forest, vector<Mat> imgs, vector<int> labels, float batches_division[], int num_trees){

    int rand_int, i, j;

    cout << "starting training" << endl;

    for(j=0;j<num_trees;j++){

        cout << "tree" << j << endl;

        Mat feats, selected_labels, tmp;

        vector<float> features;
        vector<Point> locations;

        vector<int> selected_indices;
        /*I SELECT RANDOM INDICES PER EACH CLASS, ACCORDING TO PERCENTAGES FOR EACH CLASS
        BECAUSE THERE ARE CLASSES WITH MORE IMGS THAN OTHERS*/
        getRandomIndices(labels, batches_division, &selected_indices);



        for(i=0;i<selected_indices.size();i++){
            rand_int = selected_indices[i];	//SELECT RANDOM NUMBER
            Mat tmp = imgs[rand_int];	//select img from random index

            HOGDescriptor *hog = new HOGDescriptor(
                    Size(tmp.cols, tmp.rows),		//WIN SIZE
                    Size(BLOCK_SIZE, BLOCK_SIZE),			//BLOCK SIZE
                    Size(BLOCK_STRIDE,BLOCK_STRIDE),			//BLOCK STRIDE
                    Size(CELL_SIZE,CELL_SIZE),			//CELL SIZE
                    NUM_BINS						//# BINS
            );
            hog->compute(tmp, features, Size(20,20), Size(0,0), locations);

            Mat tmp1;
            tmp1.push_back(features);
            transpose(tmp1, tmp1);

            feats.push_back(tmp1);
            selected_labels.push_back(labels[rand_int]);
        }
        feats.convertTo(feats, CV_32F);

        bool myTrainData = forest[j]->train(feats, ml::ROW_SAMPLE, selected_labels);
    }

    return forest;
}


float predict_forest(vector<Ptr<DTrees>> forest, vector<Mat> test_imgs, vector<int> labels,
                     int num_trees, int num_classes, vector<int> *predicted_values, vector<float> *perc_predicted){

    int i, j, correct = 0, wrong = 0, result_index;
    float perc;
    Mat tmp_img;

    assert(test_imgs.size() == labels.size());

    for(i=0;i<test_imgs.size();i++){
        vector<float> features;
        vector<Point> locations;

        tmp_img = test_imgs[i];
        resize(tmp_img, tmp_img, Size(160, 160));

        HOGDescriptor *hog = new HOGDescriptor(
                Size(tmp_img.cols, tmp_img.rows),
                Size(BLOCK_SIZE, BLOCK_SIZE),
                Size(BLOCK_STRIDE,BLOCK_STRIDE),
                Size(CELL_SIZE,CELL_SIZE),
                NUM_BINS
        );
        hog->compute(tmp_img, features, Size(20,20), Size(0,0), locations);

        int scores[num_classes];
        scores[num_classes] = { 0 };
        for(j=0;j<num_trees;j++){
            int prediction;
            Mat tmp3;
            tmp3.push_back(features);
            tmp3.convertTo(tmp3, CV_32F);

            prediction = forest[j]->predict(tmp3);
            scores[prediction]++;
        }
        result_index = distance(scores, max_element(scores, scores + sizeof(scores)/sizeof(scores[0])));
        predicted_values->push_back(result_index);

        perc = float(scores[result_index])/float(num_trees);
        perc_predicted->push_back(perc);

        if(labels[i] == result_index)
            correct++;
        else
            wrong++;
    }
//cout << "Correct: " << correct << " Misclassified: " << wrong << endl;

    return float(correct)/(wrong+correct);
}

#endif //TDCV_CLASSIFIEROPERATIONS_H
