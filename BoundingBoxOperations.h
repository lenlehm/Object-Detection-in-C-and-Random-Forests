//
// Created by Rajdeep Surolia on 2019-01-04.
//

#ifndef TDCV_BOUNDINGBOXOPERATIONS_H
#define TDCV_BOUNDINGBOXOPERATIONS_H

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

void arrayIndexes(vector<int>, vector<int>, vector<int>, vector<int>, vector<float>, vector<int> *, vector<int> *, vector<int> *, vector<float> *, int);
void non_max_suppression(vector<int>, vector<int>, vector<int> , vector<float>, vector<int> *);
float intersectionOverUnion(int, int, int, int, int, int, int);
void getRandomIndices(vector<int>, float [], vector<int> *);


void arrayIndexes(vector<int> original_arr, vector<int> arr1, vector<int> arr2, vector<int> arr3, vector<float> arr4,
                  vector<int> *res_1, vector<int> *res_2, vector<int> *res_3, vector<float> *res_4, int target){

    int i;

    assert(original_arr.size() == arr1.size());
    assert(original_arr.size() == arr2.size());
    assert(original_arr.size() == arr3.size());

    for(i=0;i<original_arr.size();i++){
        if(original_arr[i] == target){
            res_1->push_back(arr1[i]);
            res_2->push_back(arr2[i]);
            res_3->push_back(arr3[i]);
            res_4->push_back(arr4[i]);
        }
    }

}

void non_max_suppression(vector<int> cols, vector<int> rows, vector<int> wind_size, vector<float> scores, vector<int> *picked_indexes){

    assert(cols.size() == rows.size());
    assert(cols.size() == wind_size.size());
    assert(cols.size() == scores.size());

    if(cols.size() < 1)
        return;

    //x1 is cols, y1 is rows, x2 is cols + wind_size, y2 is rows + wind_size
    int i, j, last, pos;
    float overlap;

    //ARGSORT to take indices of ordinate score (Last index is highest score) => GREEDY APPROACH
    vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),[&scores](int i1, int i2) {return scores[i1] < scores[i2];});


    while(indices.size() > 0){
        last = indices.size() - 1;
        i = indices[last];
        picked_indexes->push_back(i);

        vector<int> suppress;
        suppress.push_back(last);

        Rect2d wind1(cols[i], rows[i], wind_size[i], wind_size[i]);

        for(pos=0;pos<last;pos++){
            j = indices[pos];
            overlap = intersectionOverUnion(cols[i], rows[i], cols[i] + wind_size[i], rows[i] + wind_size[i],
                                            cols[j], rows[j], wind_size[j]);

            Rect2d wind2(cols[j], rows[j], wind_size[j], wind_size[j]);
            Rect2d intersection = wind1 & wind2;

            if(overlap > 0.2)
                suppress.push_back(pos);
            else if(overlap > 0 && (intersection.area() == wind1.area() || intersection.area() == wind2.area()))
                suppress.push_back(pos);
        }

        //DELETE INDICES IN SUPPRESS
        sort(suppress.begin(), suppress.end());
        for(i=suppress.size()-1;i>=0;i--){
            indices.erase(indices.begin() + suppress[i]);
        }
    }

}


float intersectionOverUnion(int gt_top_x, int gt_top_y, int gt_bottom_x, int gt_bottom_y,
                            int wind_top_x, int wind_top_y, int wind_size){
    float iou, interArea;

    //SQUARE WINDOW
    assert((gt_bottom_y - gt_top_y) == (gt_bottom_x - gt_top_x));

    Rect2d wind(wind_top_x, wind_top_y, wind_size, wind_size);
    Rect2d gt_wind(gt_top_x, gt_top_y, gt_bottom_x - gt_top_x, gt_bottom_y - gt_top_y);

    Rect2d intersect = wind & gt_wind;

    //area of intersection rectangle
    interArea = intersect.area();

    //intersection area divided sum of prediction area, ground-truth, minus interesection area
    iou = interArea / float(wind.area() + gt_wind.area() - interArea);

    return iou;
}


void getRandomIndices(vector<int> labels, float percentages_batches[], vector<int> *all_indices){

    int i, j, old_lab = -1;
    int total_per_class, batch_imgs = 0;
    vector<int> labels_offset;

    for(i=0;i<labels.size();i++){
        if(old_lab != labels[i]){
            labels_offset.push_back(i);
            old_lab = labels[i];
        }
    }
    labels_offset.push_back(labels.size());

    for(i=0;i<labels_offset.size()-1;i++){
        vector<int> indices;

        for(j=labels_offset[i];j<labels_offset[i+1];j++)
            indices.push_back(j);


        srand(time(NULL));
        random_shuffle(indices.begin(), indices.end());

        total_per_class = labels_offset[i+1] - labels_offset[i];
        batch_imgs = percentages_batches[i] * total_per_class;

        for(j=0;j<batch_imgs;j++)
            all_indices->push_back(indices[j]);
    }

}

#endif //TDCV_BOUNDINGBOXOPERATIONS_H
