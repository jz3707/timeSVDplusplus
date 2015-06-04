/*
Author: Guang Yang
*/
//This file is used to generate predictions.
#include <cmath>
#include <iostream>
#include <cstring> 
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <string>
#include <string.h> 
#include <utility>
#include "SVD.cpp"

using namespace std;

int main() {
    string trainFile = "train.txt";  //set train data
    string crossFile = "cross.txt";  //set cross validation data
    string testFile = "test.txt";  //set test data
    string outFile = "5120309085_5120309016_5120309005.txt";  //set output data
    FILE *fp = fopen("training.txt","r");
    FILE *ft = fopen("train.txt","w");
    FILE *fc = fopen("cross.txt","w");
    srand(time(NULL));
    char s[2048];
    while (fscanf(fp,"%s",&s)!=EOF) {
        if (rand()%100==0) {               //use 1% random train data for cross validation
            fprintf(fc,"%s\n",s);
        }
        else {
            fprintf(ft,"%s\n",s);
        }
    }
    fclose(fp);
    fclose(ft);
    fclose(fc);
    SVD svd(NULL,NULL,0,NULL,NULL, trainFile, crossFile, testFile, outFile);
    double rmse = svd.MyTrain();     //train
    return 0;
}
