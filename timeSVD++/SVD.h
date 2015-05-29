/*
Author: Guang Yang
*/
//Header file of the class SVD.
#ifndef SVD_H_INCLUDED
#define SVD_H_INCLUDED

#include <vector>
#include <cstring>
#include <utility>
#include <algorithm>
#include <map>

using namespace std;

class SVD{
    public:
        SVD(double*,double*,int,double**,double**, string, string, string ,string);   //initialization
        virtual ~SVD();
        double MyTrain();
        virtual void Train();
        virtual double predictScore(double,int,int,int);    //prediction function
        double CalDev(int,int);    //calculate dev_u(t)
        int CalBin(int);    //calculate time bins
        double Validate(double,double*,double*,double**,double**);    //validation function
    protected:

        //   prediction formula:
        //   avg + Bu + Bi
        //   + Bi_Bin,t + Alpha_u*Dev + Bu_t
        //   + Qi^T(Pu + |R(u)|^-1/2 \sum yi

        double* Tu;           //variable for mean time of user
        double* Alpha_u;
        double* Bi;
        double** Bi_Bin;
        double* Bu;
        vector<map<int,double> > Bu_t;
        vector<map<int,double> > Dev;    //save the result of CalDev(userId,time)
        double** Qi;
        double** Pu;
        double** y;
        double** sumMW;    //save the sum of Pu
        string trainFile;
        string crossFile;
        string testFile;
        string outFile;
        vector <vector<pair <pair<int,int>, int> > > train_data;
        vector <pair <pair<int,int>, pair <int, int> > > test_data;
 };


#endif // SVD_H_INCLUDED
