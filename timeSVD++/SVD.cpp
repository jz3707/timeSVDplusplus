/*
Author: Guang Yang
*/
//This file is the implementation of the class SVD.
#include <cmath>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cassert>
#include <ctime>
#include <utility>
#include "SVD.h"
#define sign(n) (n==0? 0 : (n<0?-1:1))    //define sign function

using namespace std;

const int userNum = 100000;  //number of users
const int itemNum = 17770;   //number of items
const int timeNum = 5115;    //number of days(time)
const int binNum = 30;       //number of time bins
const double AVG = 3.60073;  //average score
double G_alpha = 0.00001;        //gamma for alpha
const double L_alpha = 0.0004;   //learning rate for alpha
const double L_pq = 0.015;       //learning rate for Pu & Qi
double G = 0.007;                //general gamma
const double Decay = 0.9;        //learning rate decay factor
const double L = 0.005;          //general learning rate
const int factor = 50;           //number of factors

//calculate dev_u(t) = sign(t-tu)*|t-tu|^0.4 and save the result for saving the time
double SVD::CalDev(int user, int timeArg) {
    if(Dev[user].count(timeArg)!=0)return Dev[user][timeArg];
    double tmp = sign(timeArg - Tu[user]) * pow(double(abs(timeArg - Tu[user])), 0.4);
    Dev[user][timeArg] = tmp;
    return tmp;
}

//calculate time bins
int SVD::CalBin(int timeArg) {
    int binsize = timeNum/binNum + 1;
    return timeArg/binsize;
}

//main function for training
//terminate when RMSE varies less than 0.00005
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
        if(curRmse >= preRmse-0.00005){
            break;
        }
        else{
            preRmse = curRmse;
        }
    }
    while (fscanf(fp,"%d,%d,%d",&user, &item, &date)!=EOF) {
        fout << predictScore(AVG,user,item,date) << endl;
    }
    fclose(fp);
    fout.close();
    return curRmse;
}


//function for cross validation
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
        double pScore = predictScore(avg,userId,itemId,t);
        rmse += (rating - pScore) * (rating - pScore);
    }
    return sqrt(rmse/n);
}


//function for prediction
//   prediction formula:
//   avg + Bu + Bi
//   + Bi_Bin,t + Alpha_u*Dev + Bu_t
//   + Qi^T(Pu + |R(u)|^-1/2 \sum yi
double SVD::predictScore(double avg,int userId, int itemId,int time){
    double tmp = 0.0;
    int sz = train_data[userId].size();
    double sqrtNum = 0;
    if (sz>1) sqrtNum = 1/(sqrt(sz));
    for(size_t i=0;i<factor;i++){
        tmp += (Pu[userId][i] +sumMW[userId][i]*sqrtNum) * Qi[itemId][i];
    }
    double score = avg + Bu[userId] + Bi[itemId] + Bi_Bin[itemId][CalBin(time)] + Alpha_u[userId]*CalDev(userId,time) + Bu_t[userId][time] + tmp;

    if(score > 5){
        score = 5;
    }
    if(score < 1){
        score = 1;
    }
    return score;
}

//function for training
//update using stochastic gradient descent

void SVD::Train(){
    int userId,itemId,rating,time;
    for (userId = 0; userId < userNum; ++userId) {
        int sz = train_data[userId].size();
        double sqrtNum = 0;
        vector <double> tmpSum(factor,0);
        if (sz>1) sqrtNum = 1/(sqrt(sz));
        for (int k = 0; k < factor; ++k) {
            double sumy = 0;
            for (int i = 0; i < sz; ++i) {
                int itemI = train_data[userId][i].first.first;
                sumy += y[itemI][k];
            }
            sumMW[userId][k] = sumy;
        }
        for (int i = 0; i < sz; ++i) {
            itemId = train_data[userId][i].first.first;
            rating = train_data[userId][i].first.second;
            time = train_data[userId][i].second;
            double predict = predictScore(AVG,userId,itemId,time);
            double error = rating - predict;
            Bu[userId] += G * (error - L * Bu[userId]);
            Bi[itemId] += G * (error - L * Bi[itemId]);
            Bi_Bin[itemId][CalBin(time)] += G * (error - L * Bi_Bin[itemId][CalBin(time)]);
            Alpha_u[userId] += G_alpha * (error * CalDev(userId,time)  - L_alpha * Alpha_u[userId]);
            Bu_t[userId][time] += G * (error - L * Bu_t[userId][time]);

            for(size_t k=0;k<factor;k++){
                auto uf = Pu[userId][k];
                auto mf = Qi[itemId][k];
                Pu[userId][k] += G * (error * mf - L_pq * uf);
                Qi[itemId][k] += G * (error * (uf+sqrtNum*sumMW[userId][k]) - L_pq * mf);
                tmpSum[k] += error*sqrtNum*mf;
            }
        }
        for (int j = 0; j < sz; ++j) {
            itemId = train_data[userId][j].first.first;
            for (int k = 0; k < factor; ++k) {
                double tmpMW = y[itemId][k];
                y[itemId][k] += G*(tmpSum[k]- L_pq *tmpMW);
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
                int itemI = train_data[userId][i].first.first;
                sumy += y[itemI][k];
            }
            sumMW[userId][k] = sumy;
        }
    }
    G *= Decay;
    G_alpha *= Decay;
}

//initialization

SVD::SVD(double* bi,double* bu,int k,double** qi,double** pu, string train_file, string cross_file, string test_file, string out_file):
 trainFile(train_file), crossFile(cross_file), testFile(test_file), outFile(out_file){
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

    Alpha_u = new double[userNum];
    for(size_t i=0;i<userNum;i++){
        Alpha_u[i] = 0;
    }

    Bi_Bin = new double* [itemNum];
    for(size_t i=0;i<itemNum;i++){
        Bi_Bin[i] = new double[binNum];
    }

    for(size_t i=0;i<itemNum;i++){
        for(size_t j=0;j<binNum;j++){
            Bi_Bin[i][j] = 0.0;
        }
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
            }
        }
    }   else{
        Pu = pu;
    }
    FILE *fp = fopen(trainFile.c_str(),"r");
    int userId,itemId,rating,t;
    while(fscanf(fp,"%d,%d,%d,%d",&userId, &itemId, &t, &rating)!=EOF){
        train_data[userId].push_back(make_pair(make_pair(itemId,rating),t));
    }
    fclose(fp);
    fp = fopen(crossFile.c_str(),"r");
    while(fscanf(fp,"%d,%d,%d,%d",&userId, &itemId, &t, &rating)!=EOF){
        test_data.push_back(make_pair(make_pair(userId, itemId),make_pair(t,rating)));
    }
    fclose(fp);

    Tu = new double[userNum];
    for(size_t i=0;i<userNum;i++){
        double tmp = 0;
        if(train_data[i].size()==0)
        {
            Tu[i] = 0;
            continue;
        }
        for(size_t j=0;j<train_data[i].size();j++){
            tmp += train_data[i][j].second;
        }
        Tu[i] = tmp/train_data[i].size();
    }

    for(size_t i=0;i<userNum;i++){
        map<int,double> tmp;
        for(size_t j=0;j<train_data[i].size();j++){
            if(tmp.count(train_data[i][j].second)==0)
            {
                tmp[train_data[i][j].second] = 0.0000001;
            }
            else continue;
        }
        Bu_t.push_back(tmp);
    }

    for(size_t i=0;i<userNum;i++){
        map<int,double> tmp;
        Dev.push_back(tmp);
    }

}

SVD::~SVD(){
    delete[] Bi;
    delete[] Bu;
    delete[] Alpha_u;
    delete[] Tu;
    for(size_t i=0;i<userNum;i++){
        delete[] Pu[i];
        delete[] sumMW[i];
    }
    for(size_t i=0;i<itemNum;i++){
        delete[] Bi_Bin[i];
        delete[] Qi[i];
        delete[] y[i];
    }
    delete[] Bi_Bin;
    delete[] sumMW;
    delete[] y;
    delete[] Pu;
    delete[] Qi;
}
