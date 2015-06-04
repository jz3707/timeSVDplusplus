/*
Author: Zebing Lin
Email:linzebing1995@gmail.com
*/
//This file the implementation of the class SVD.
#include <cmath>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <utility>
#include "SVD.h"

using namespace std;

const int userNum = 100000;
const int itemNum = 17770;
const double AVG = 3.60073;

double SVD::MyTrain() {
    double preRmse = 1000;
    ofstream fout(outFile.c_str());
    srand(time(NULL));
    FILE *fp = fopen(testFile.c_str(),"r");
    int user, item, date;
    double curRmse;
    for(size_t i=0;i<1000;i++) {
        Train();
        curRmse = Validate(AVG,Bu,Bi,Pu,Qi);
        cout << "test_Rmse in step " << i << ": " << curRmse << endl;
        if(curRmse >= preRmse-0.000005){
            break;
        }
        else{
            preRmse = curRmse;
        }
    }
    while (fscanf(fp,"%d,%d,%d",&user, &item, &date)!=EOF) {
        fout << predictScore(AVG,Bu[user],Bi[item], Pu[user],Qi[item]) << endl;
    }
    fclose(fp);
    fout.close();
    return curRmse;
}

double SVD::Validate(double avg,double* bu,double* bi,double** pu,double** qi){
    int userId,itemId,rating,t;
    int n = 0;
    double rmse = 0;
    for (const auto &ch:test_data){
        userId = ch.first.first;
        itemId = ch.first.second;
        t = ch.second.first;
        rating = ch.second.second;
        ++n;
        double pScore = predictScore(avg,bu[userId],bi[itemId],pu[userId],qi[itemId]);
        rmse += (rating - pScore) * (rating - pScore);
    }
    return sqrt(rmse/n);
}

double SVD::predictScore(double avg,double bu,double bi,double* pu,double* qi){
    double tmp = 0.0;
    for(size_t i=0;i<factor;i++){
        tmp += pu[i] * qi[i];
    }
    double score = avg + bu + bi + tmp;
    if(score > 5){
        score = 5;
    }
    if(score < 1){
        score = 1;
    }
    return score;
}

void SVD::Train(){
    int userId,itemId,rating,t;
    for (const auto &ch:train_data) {
        userId = ch.first.first;
        itemId = ch.first.second;
        t = ch.second.first;
        rating = ch.second.second;
        double predict = predictScore(AVG,Bu[userId],Bi[itemId],Pu[userId],Qi[itemId]);
        double error = rating - predict;
        Bu[userId] += lr * (error - theta * Bu[userId]);
        Bi[itemId] += lr * (error - theta * Bi[itemId]);

        for(size_t i=0;i<factor;i++){
            double Tmp = Pu[userId][i];
            Pu[userId][i] += lr * (error * Qi[itemId][i] - theta * Pu[userId][i]);
            Qi[itemId][i] += lr * (error * Tmp - theta * Qi[itemId][i]);
        }
    }
}

SVD::SVD(double* bi,double* bu,int k,double** qi,double** pu, string train_file, string cross_file, string test_file, string out_file, double lrate, double th, int fac):
lr(lrate),theta(th),factor(fac), trainFile(train_file), crossFile(cross_file), testFile(test_file), outFile(out_file){
    if(bi == NULL){
        Bi = new double[itemNum];
        for(size_t i=0;i<itemNum;i++){
            Bi[i] = 0.0;
        }
    }
    else{
        Bi = bi;
    }

    if(bu == NULL){
        Bu = new double[userNum];
        for(size_t i=0;i<userNum;i++){
            Bu[i] = 0.0;
        }
    }
    else{
        Bu = bu;
    }

    if(qi == NULL){
        Qi = new double* [itemNum];
        for(size_t i=0;i<itemNum;i++){
            Qi[i] = new double[factor];
        }

        for(size_t i=0;i<itemNum;i++){
            for(size_t j=0;j<factor;j++){
                Qi[i][j] = Qi[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
            }
        }
    }
    else{
        Qi = qi;
    }

    if(pu == NULL){
        Pu = new double* [userNum];
        for(size_t i=0;i<userNum;i++){
            Pu[i] = new double[factor];
        }

        for(size_t i=0;i<userNum;i++){
            for(size_t j=0;j<factor;j++){
                Pu[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
            }
        }
    }   else{
        Pu = pu;
    }
    FILE *fp = fopen(trainFile.c_str(),"r");
    int userId,itemId,rating,t;
    while(fscanf(fp,"%d,%d,%d,%d",&userId, &itemId, &t, &rating)!=EOF){
        train_data.push_back(make_pair(make_pair(userId, itemId),make_pair(t,rating)));
    }
    fclose(fp);
    fp = fopen(crossFile.c_str(),"r");
    while(fscanf(fp,"%d,%d,%d,%d",&userId, &itemId, &t, &rating)!=EOF){
        test_data.push_back(make_pair(make_pair(userId, itemId),make_pair(t,rating)));
    }
    fclose(fp);
}

SVD::~SVD(){
    delete[] Bi;
    delete[] Bu;
    for(size_t i=0;i<userNum;i++){
        delete[] Pu[i];
    }
    for(size_t i=0;i<itemNum;i++){
        delete[] Qi[i];
    }
    delete[] Pu;
    delete[] Qi;
}
