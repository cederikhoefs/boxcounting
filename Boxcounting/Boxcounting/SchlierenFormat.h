#pragma once
#include "stdafx.h"

using namespace std;

typedef uint16_t iter_t;

class SchlierenFile {
public:
	static const char Magic[];
	uint32_t Resolution;
	uint32_t MaxIteration;
	double Scale;
	double Viewport_x;
	double Viewport_y;

	iter_t* data;

	SchlierenFile();
	SchlierenFile(ifstream& inputfile);
	SchlierenFile(uint32_t res, uint32_t maxiter, float scale, float vx, float vy);
	~SchlierenFile();

	void write(ofstream& outfile);
	void toPNG(string filename, int k);
};

ostream& operator << (ostream& os, SchlierenFile& f);
