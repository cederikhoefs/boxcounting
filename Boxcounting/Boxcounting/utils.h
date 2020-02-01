#pragma once

#include "stdafx.h"

struct Color
{
	uint8_t R, G, B, A;
};

extern Color white;
extern Color black;

extern std::string clErrInfo(cl::Error e);
extern void printDevice(int i, cl::Device& d);
extern void printPlatform(int i, cl::Platform& p);
extern void print2D(uint8_t* buffer, int res);
extern void drawPNG(uint8_t* buffer, int res, std::string filename, Color yes = black, Color no = white);

