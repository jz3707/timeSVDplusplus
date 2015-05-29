/*
Author: Zebing Lin
Email:linzebing1995@gmail.com
*/
//Implementation of the class SVD.
#include <cmath>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cassert>
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
        fout << predictScore(AVG,user,item) << endl;
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
        n++;
        double pScore = predictScore(avg,userId,itemId);
        rmse += (rating - pScore) * (rating - pScore);
    }
    return sqrt(rmse/n);
}

double SVD::predictScore(double avg,int userId, int itemId){
    double tmp = 0.0;
    int sz = train_data[userId].size();
    double sqrtNum = 0;
    if (sz>1) sqrtNum = 1/(sqrt(sz));
    for(size_t i=0;i<factor;i++){
        tmp += (Pu[userId][i]+sumMW[userId][i]*sqrtNum) * Qi[itemId][i];
    }
    double score = avg + Bu[userId] + Bi[itemId] + tmp;
    //assert(!isnan(score));
    if(score > 5){
        score = 5;
    }
    if(score < 1){
        score = 1;
    }
    return score;
}

void SVD::Train(){
    int userId,itemId,rating;
    for (userId = 0; userId < userNum; ++userId) {
        int sz = train_data[userId].size();
        double sqrtNum = 0;
        vector <double> tmpSum(factor,0);
        if (sz>1) sqrtNum = 1/(sqrt(sz));
        for (int k = 0; k < factor; ++k) {
            double sumy = 0;
            for (int i = 0; i < sz; ++i) {
                int itemI = train_data[userId][i].first;
                sumy += y[itemI][k];
            }
            sumMW[userId][k] = sumy;
        }
        for (int i = 0; i < sz; ++i) {
            itemId = train_data[userId][i].first;
            rating = train_data[userId][i].second;
            double predict = predictScore(AVG,userId,itemId);
            double error = rating - predict;
            Bu[userId] += lr * (error - theta * Bu[userId]);
            Bi[itemId] += lr * (error - theta * Bi[itemId]);

            for(size_t k=0;k<factor;k++){
                auto uf = Pu[userId][k];
                auto mf = Qi[itemId][k];
                Pu[userId][k] += lr * (error * mf - 0.015 * uf);
                Qi[itemId][k] += lr * (error * (uf+sqrtNum*sumMW[userId][k]) - 0.015 * mf);
                tmpSum[k] += error*sqrtNum*mf;
            }
        }
        for (int j = 0; j < sz; ++j) {
            itemId = train_data[userId][j].first;
            for (int k = 0; k < factor; ++k) {
                double tmpMW = y[itemId][k];
                y[itemId][k] += lr*(tmpSum[k]-0.015*tmpMW);
                sumMW[userId][k] += y[itemId][k] - tmpMW;
            }
        }
    }
    for (userId = 0; userId < userNum; ++userId) {
        auto sz = train_data[userId].size();
        double sqrtNum = 0;
        if (sz>1) sqrtNum = 1.0/sqrt(sz);
        for (int k = 0; k < factor; ++k) {
            double sumy = 0;
            for (int i = 0; i < sz; ++i) {
                int itemI = train_data[userId][i].first;
                sumy += y[itemI][k];
            }
            sumMW[userId][k] = sumy;
        }
    }
    lr *= (0.9+0.1*rand()/RAND_MAX);
}

SVD::SVD(double* bi,double* bu,int k,double** qi,double** pu, string train_file, string cross_file, string test_file, string out_file, double lrate, double th, int fac):
lr(lrate),theta(th),factor(fac), trainFile(train_file), crossFile(cross_file), testFile(test_file), outFile(out_file){
    train_data.resize(userNum);
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
        y = new double* [itemNum];
        for(size_t i=0;i<itemNum;i++){
            Qi[i] = new double[factor];
            y[i] = new double[factor];
        }

        for(size_t i=0;i<itemNum;i++){
            for(size_t j=0;j<factor;j++){
                Qi[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
                y[i][j] = 0;
            }
        }
    }
    else{
        Qi = qi;
    }

    if(pu == NULL){
        sumMW = new double* [userNum];
        Pu = new double* [userNum];
        for(size_t i=0;i<userNum;i++){
            Pu[i] = new double[factor];
            sumMW[i] = new double[factor];
        }

        for(size_t i=0;i<userNum;i++){
            for(size_t j=0;j<factor;j++){
                sumMW[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
                Pu[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
                //Pu[i][j] = 0;
            }
        }
    }   else{
        Pu = pu;
    }
    FILE *fp = fopen(trainFile.c_str(),"r");
    int userId,itemId,rating,t;
    while(fscanf(fp,"%d,%d,%d,%d",&userId, &itemId, &t, &rating)!=EOF){
        train_data[userId].push_back(make_pair(itemId,rating));
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
        delete[] sumMW[i];
    }
    for(size_t i=0;i<itemNum;i++){
        delete[] Qi[i];
        delete[] y[i];
    }
    delete[] sumMW;
    delete[] y;
    delete[] Pu;
    delete[] Qi;
}
