#ifndef SVD_H_INCLUDED
#define SVD_H_INCLUDED
/*
Author: Zebing Lin
Email:linzebing1995@gmail.com
*/
//Header file of the class SVD.
#include <vector>
#include <cstring>
#include <utility>

using namespace std;

class SVD{
    public:
        SVD(double*,double*,int,double**,double**, string, string, string ,string, double , double , int);
        virtual ~SVD();
        double MyTrain();
        virtual void Train();
        virtual double predictScore(double,double,double,double*,double*);
        double Validate(double,double*,double*,double**,double**);
    protected:
        double* Bi;
        double* Bu;
        int factor;
        double** Qi;
        double** Pu;
        double lr;
        double theta;
        string trainFile;
        string crossFile;
        string testFile;
        string outFile;
        vector <pair <pair<int,int>, pair <int, int> > > train_data;
        vector <pair <pair<int,int>, pair <int, int> > > test_data;
 };


#endif // SVD_H_INCLUDED
