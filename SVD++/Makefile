all: run.sh
clear:
	rm *.o
run.sh: main.o
	g++ main.o -o run.sh
main.o: main.cpp SVD.h
	g++ -std=c++0x -c main.cpp

