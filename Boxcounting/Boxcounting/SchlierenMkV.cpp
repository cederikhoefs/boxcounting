#include "stdafx.h"
#include "utils.h"

//#define STD_EXPORT
#define CSV_EXPORT
#define PNG_TILE_EXPORT
#undef CSV_APPEND

using namespace std;
using namespace std::chrono;

cl::Device device;
cl::Context context;
cl::Program program;
cl::CommandQueue queue;

//const int log2res = 7;
//const int64_t Resolution = (1 << log2res);

const int TileResolution = 4096;
const int TileCount = 1;
const double Scale = 3.0;
const int Iteration = 1000;
const double Viewport_x = 1.0;
const double Viewport_y = 0.0;

bool initOpenCL(cl::Device& device, cl::Context& context, cl::Program& prog, cl::CommandQueue& q)
{

	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	int platId = 0;

	if (platforms.size() == 0)
		throw string("No devices found");
	for (int i = 0; i < platforms.size(); i++)
		printPlatform(i, platforms[i]);

	if (platforms.size() != 1) {
		cout << "Platform choice (0 - " << platforms.size() - 1 << "): ";
		cin >> platId;
	}
	else {
		cout << "Choosing the only platform" << endl;
	}
	if (platId < 0 || platId >= platforms.size())
		throw string("Invalid platform choice");

	cl::Platform platform = platforms[platId];

	cout << "OpenCL version: " << platform.getInfo<CL_PLATFORM_VERSION>() << endl;

	if (platform() == 0)
		throw string("No OpenCL 2.0 platform found");

	vector<cl::Device> gpus;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
	if (gpus.size() == 0)
		throw string("No devices found");
	for (int i = 0; i < gpus.size(); i++)
		printDevice(i, gpus[i]);

	unsigned int deviceId = 0;
	if (gpus.size() != 1) {
		cout << "Device choice (0 - " << gpus.size() - 1 << "): ";
		cin >> deviceId;
	}
	else {
		cout << "Choosing the only GPU" << endl;
	}
	if (deviceId < 0 || deviceId >= gpus.size())
		throw string("Invalid device choice");

	device = gpus[deviceId];

	cout << "Creating context... " << endl;
	context = cl::Context({ device });

	cout << "Compiling sources... " << endl;
	cl::Program::Sources sources;
	ifstream sourcefile("kernel.cl");
	string sourcecode(istreambuf_iterator<char>(sourcefile), (istreambuf_iterator<char>()));
	sources.push_back(sourcecode);

	prog = cl::Program(context, sources);
	try {
		prog.build({ device });
	}
	catch (cl::Error e) {
		throw string("OpenCL build error:\n") + prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	}
	cout << "Creating command queue" << endl;
	q = cl::CommandQueue(context, device);

	cout << "InitOpenCL finished!" << endl;

	return true;
}

void calculate(uint8_t* schlieren, int32_t res, int32_t iter, double scale = 6.0, double vx = 0.0, double vy = 0.0)
{
	cl::Buffer schlierenbuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * res * res);

	cl::Kernel kernel = cl::Kernel(program, "mandelbrot");
	kernel.setArg(0, schlierenbuffer);
	kernel.setArg(1, scale);
	kernel.setArg(2, res);
	kernel.setArg(3, iter);
	kernel.setArg(4, vx);
	kernel.setArg(5, vy);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(res * res), cl::NullRange);
	queue.finish();
	queue.enqueueReadBuffer(schlierenbuffer, CL_TRUE, 0, sizeof(uint8_t) * res * res, schlieren);
}

void scaledown(uint8_t* schlieren_old, uint8_t* schlieren_new, int oldres)
{
	int newres = oldres / 2;
	cl::Buffer oldbuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * oldres * oldres);
	cl::Buffer newbuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * newres * newres);

	cl::Kernel kernel = cl::Kernel(program, "scaledown");

	kernel.setArg(0, oldbuffer);
	kernel.setArg(1, newbuffer);
	kernel.setArg(2, oldres);

	queue.enqueueWriteBuffer(oldbuffer, CL_FALSE, 0, sizeof(uint8_t) * oldres * oldres, schlieren_old);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(newres * newres), cl::NullRange/*cl::NDRange(newres)*/);
	queue.finish();
	queue.enqueueReadBuffer(newbuffer, CL_TRUE, 0, sizeof(uint8_t) * newres * newres, schlieren_new);
}

int sumup(uint8_t* schlieren, int res)
{
	cl::Kernel firstredux = cl::Kernel(program, "firstsum");
	cl::Kernel redux = cl::Kernel(program, "sum");

	cl::Buffer origin(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * res * res);
	cl::Buffer buf1(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * res/2 * res/2);
	cl::Buffer buf2(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * res/4 * res/4);

	queue.enqueueWriteBuffer(origin, CL_FALSE, 0, sizeof(uint8_t) * res * res, schlieren);

	firstredux.setArg(0, res);
	firstredux.setArg(1, origin);
	firstredux.setArg(2, buf1);
	queue.enqueueNDRangeKernel(firstredux, cl::NullRange, cl::NDRange(res/2 * res/2));
	queue.finish();

	cl::Buffer *from = &buf1, *to = &buf2;

	res /= 2;
	for (; res > 1; res = res/2) {
		redux.setArg(0, res);
		redux.setArg(1, *from);
		redux.setArg(2, *to);
		queue.enqueueNDRangeKernel(redux, cl::NullRange, cl::NDRange(res/2 * res/2));
		queue.finish();
		swap(from, to);
	}

	uint32_t result = -1;
	queue.enqueueReadBuffer(*from, CL_TRUE, 0, sizeof(uint32_t) * 1, &result);
	return result;
}

int cpu_sumup(uint8_t* schlieren, int res)
{
	int sum = 0;
	for (int i = 0; i < res; i++)
		for (int j = 0; j < res; j++)
			sum += schlieren[j * res + i];

	return sum;
}

void testscaledown()
{
	uint8_t* A = new uint8_t[4096 * 4096];
	uint8_t* B = new uint8_t[8192 * 8192];

	cout << "Calculating 1024 x 1024...";

	try {
		calculate(A, 1024, Iteration, Scale, Viewport_x, Viewport_y);
	}
	catch (cl::Error e) {
		cout << clErrInfo(e) << endl;
	}

	cout << " finished: " << sumup(A, 1024) << endl;

	//cout << "Exporting PNG...";
	//drawPNG(A, 128, "128.png");
	//cout << " finished." << endl;

	cout << "Calculating 8192 x 8192...";

	try {
		calculate(B, 8192, Iteration, Scale, Viewport_x, Viewport_y);
	}
	catch (cl::Error e) {
		cout << clErrInfo(e) << endl;
	}

	cout << " finished: " << endl;

	cout << "Scaling down n times...";
	scaledown(B, A, 8192);
	scaledown(A, B, 4096);
	scaledown(B, A, 2048);

	cout << "Sumup: " << sumup(A, 1024) << endl;

	//cout << "Exporting PNG...";
	//drawPNG(B, 128, "8192scaleddown.png");
	//cout << " finished." << endl;

	delete[] A, B;
}

template<int TileSize = 1024>
vector<uint32_t> tiling(uint32_t resintiles = 16, int32_t iteration = 1000, double scale = 6.0, double vx = 0.0, double vy = 0.0) {

	const int log2tilesize = log2(TileSize);

	uint8_t* TileA = new uint8_t[TileSize * TileSize];
	uint8_t* TileB = new uint8_t[TileSize * TileSize];

	uint8_t* Tiles[] = { TileA, TileB };

	vector<uint32_t> Sums(log2tilesize);

	for (int j = 0; j < resintiles; j++) {
		for (int i = 0; i < resintiles; i++) {

			int ActualRes = TileSize;

			double tilevx = ((double)i / resintiles - 0.5) * scale + scale / (resintiles) / 2 + vx;
			double tilevy = (0.5 - (double)j / resintiles) * scale - scale / (resintiles) / 2 + vy;
			double tilescale = scale / resintiles;

			cout << "Computing (" << i + 1<< ";"  << j + 1<< ")" << " of (" << resintiles << ";"  << resintiles << ")" <<  " Tiles with V(" << tilevx << ";" << tilevy << "), S = " << tilescale << endl;

			try {
				calculate(TileA, TileSize, iteration, tilescale, tilevx, tilevy);
			}
			catch (cl::Error e) {
				cout << clErrInfo(e) << endl;
				exit(-1);
			}

#ifdef PNG_TILE_EXPORT
			drawPNG(TileA, TileSize, "k="+to_string(iteration) + "tile=" + to_string(TileSize) + "(" + to_string(i) + ";" + to_string(j) + ").png");
#endif

			for (int d = 0; d < log2tilesize; d++) {

				Sums[d] += sumup(Tiles[d % 2], ActualRes);

				//cout << "Downscaling " << d + 1 << " of " << log2tilesize << endl;

				try {
					scaledown(Tiles[d % 2], Tiles[(d + 1) % 2], ActualRes);
				}
				catch (cl::Error e) {
					cout << clErrInfo(e) << endl;
					exit(-1);
				}

				ActualRes /= 2;

			}

		}

	}

	for (int i = 0; i < Sums.size(); i++)
		cout << Sums[i] << ";";


	return Sums;
}

int main(int argc, char* argv[])
{
#ifdef CSV_EXPORT
#ifdef CSV_APPEND
	ofstream outfile("dim.csv", ios::out | ios::app);
#else
	ofstream outfile("dim.csv", ios::out);
#endif
#endif
	try {
		initOpenCL(device, context, program, queue);
	}
	catch (string s) {
		cout << s << endl << "Could not init OpenCL." << endl;
		return -1;
	}


	cout << "Starting computation" << endl;
	//testscaledown();

	auto t0 = std::chrono::high_resolution_clock::now();
	vector<uint32_t> Result = tiling<TileResolution>(TileCount, Iteration, Scale, Viewport_x, Viewport_y);
	auto t1 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	cout << "Finished after " << duration << "ms" << endl;

#ifdef CSV_EXPORT
#ifndef CSV_APPEND
	outfile << "S;B;k;r;N;log r;log N" << endl;
#endif // !CSV_APPEND



	for (int i = 0; i < Result.size(); i++) {
		outfile << Scale << ";";
		outfile << (TileResolution >> i) * TileCount << ";";
		outfile << Iteration << ";";
		outfile << (TileResolution >> i) * TileCount / Scale << ";";
		outfile << Result[i] << ";";
		outfile << log10((TileResolution >> i)* TileCount / Scale) << ";";
		outfile << log10(Result[i]);
		outfile << endl;

	}

	outfile.close();
#endif // CSV_EXPORT

	system("PAUSE");
	return 0;
}

