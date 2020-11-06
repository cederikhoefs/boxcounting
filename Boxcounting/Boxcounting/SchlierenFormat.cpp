#include "stdafx.h"
using namespace std;

const char SchlierenFile::Magic[] = "VIETABOX";

SchlierenFile::SchlierenFile()
{
}

SchlierenFile::SchlierenFile(ifstream& inputfile)
{
	inputfile.seekg(0);
	
	char rawmagic[8];
	inputfile.read(rawmagic, 8);

	string magicstring(rawmagic, 8);
	if (magicstring != Magic) {
		cout << "Invalid magic string in inputfile!" << endl;
		exit(-1);
	}

	inputfile.read((char*)(&Resolution), sizeof(Resolution));
	inputfile.read((char*)(&MaxIteration), sizeof(MaxIteration));
	inputfile.read((char*)(&Scale), sizeof(Scale));
	inputfile.read((char*)(&Viewport_x), sizeof(Viewport_x));
	inputfile.read((char*)(&Viewport_y), sizeof(Viewport_y));

	data = new iter_t[Resolution * Resolution];
	inputfile.read((char*)data, (sizeof(iter_t) * (uint64_t)Resolution * Resolution));

}

SchlierenFile::SchlierenFile(uint32_t res, uint32_t maxiter, float scale, float vx, float vy)
	: Resolution(res), MaxIteration(maxiter), Scale(scale), Viewport_x(vx), Viewport_y(vy)
{
	data = new iter_t[Resolution * Resolution];
}

SchlierenFile::~SchlierenFile()
{
	delete[] data;
}

void SchlierenFile::write(ofstream& outputfile)
{
	outputfile.seekp(0);
	outputfile.write(Magic, 8);
	outputfile.write((char*)(&Resolution), sizeof(Resolution));
	outputfile.write((char*)(&MaxIteration), sizeof(MaxIteration));
	outputfile.write((char*)(&Scale), sizeof(Scale));
	outputfile.write((char*)(&Viewport_x), sizeof(Viewport_x));
	outputfile.write((char*)(&Viewport_y), sizeof(Viewport_y));
	outputfile.write((char*)data, (uint64_t)Resolution * Resolution * sizeof(iter_t));

}

void SchlierenFile::toPNG(string filename, int k)
{
	cout << "Exporting PNG with k <= " << k << endl;
	vector<uint8_t> Image(4 * Resolution * Resolution);
	bool black;

	for (int i = 0; i < Resolution * Resolution; i++) {

		//black = data[i];
		black = (data[i] <= k);
		if (black) {
			Image[4 * i + 0] = 0;
			Image[4 * i + 1] = 0;
			Image[4 * i + 2] = 0;
			Image[4 * i + 3] = 255;
		}
		else {
			Image[4 * i + 0] = 255;
			Image[4 * i + 1] = 255;
			Image[4 * i + 2] = 255;
			Image[4 * i + 3] = 255;
		}
	}

	unsigned error = lodepng::encode(filename, Image, Resolution, Resolution);

	if (error)
		std::cout << "LodePNG error: " << error << ": " << lodepng_error_text(error) << std::endl;
	}

ostream& operator << (ostream& os, SchlierenFile& f) {
	os << "[" << f.Resolution << "x" << f.Resolution << "]" \
		<< " = [" << f.Scale << "x" << f.Scale << "] @(" << f.Viewport_x << "|" << f.Viewport_y << "); k <= " << f.MaxIteration << endl;
	return os;
}