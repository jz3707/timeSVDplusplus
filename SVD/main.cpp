/*
Author: Zebing Lin
Email:linzebing1995@gmail.com
*/
//This file is used set parameters and generate predictions.
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
    FILE *fp = fopen("training.txt","r");
    FILE *ft = fopen("train.txt","w");
    FILE *fc = fopen("cross.txt","w");
    srand(time(NULL));
    char s[2048];
    while (fscanf(fp,"%s",&s)!=EOF) {
        if (rand()%100==0) {
            fprintf(fc,"%s\n",s);
        }
        else {
            fprintf(ft,"%s\n",s);
        }
    }
    fclose(fp);
    fclose(ft);
    fclose(fc);
    double lr = 0.005;
    double theta = 0.02;
    int factor = 200;
    string trainFile = "train.txt";
    string crossFile = "cross.txt";
    string testFile = "test.txt";
    string outFile = "5120309085_5120309016_5120309005.txt";
    SVD svd(NULL,NULL,0,NULL,NULL, trainFile, crossFile, testFile, outFile, lr, theta, factor);
    double rmse = svd.MyTrain();
    return 0;
}
