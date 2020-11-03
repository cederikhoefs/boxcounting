#include "stdafx.h"
#include "utils.h"

using namespace std;
using namespace std::chrono;


cl::Device device;
cl::Context context;
cl::Program program;
cl::CommandQueue queue;

vector<uint32_t> Factors;
uint32_t Resolution;
uint32_t MaxIteration;
const double Scale = 6.0;
const double Viewport_x = 0.0;
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

	//vector<cl::Device> gpus;
	//platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
	//if (gpus.size() == 0)
	//	throw string("No devices found");
	//for (int i = 0; i < gpus.size(); i++)
	//	printDevice(i, gpus[i]);

	//unsigned int deviceId = 0;
	//if (gpus.size() != 1) {
	//	cout << "Device choice (0 - " << gpus.size() - 1 << "): ";
	//	cin >> deviceId;
	//}
	//else {
	//	cout << "Choosing the only GPU" << endl;
	//}
	vector<cl::Device> devs;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devs);
	if (devs.size() == 0)
		throw string("No devices found");
	for (int i = 0; i < devs.size(); i++)
		printDevice(i, devs[i]);

	unsigned int deviceId = 0;
	if (devs.size() != 1) {
		cout << "Device choice (0 - " << devs.size() - 1 << "): ";
		cin >> deviceId;
	}
	else {
		cout << "Choosing the only GPU" << endl;
	}
	if (deviceId < 0 || deviceId >= devs.size())
		throw string("Invalid device choice");

	device = devs[deviceId];

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

void generate_schlieren(SchlierenFile& s)
{
	//cout << "Generating " << "[" << s.Resolution << "x" << s.Resolution << "] buffer with k <= " << s.MaxIteration << endl;

	cl::Buffer schlierenbuffer(context, CL_MEM_READ_WRITE, sizeof(iter_t) * s.Resolution * s.Resolution);

	cl::Kernel kernel = cl::Kernel(program, "schlieren");
	kernel.setArg(0, schlierenbuffer);
	kernel.setArg(1, (double)s.Scale);
	kernel.setArg(2, (cl_int)s.Resolution);
	kernel.setArg(3, (cl_int)s.MaxIteration);
	kernel.setArg(4, (double)s.Viewport_x);
	kernel.setArg(5, (double)s.Viewport_y);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(s.Resolution * s.Resolution), cl::NullRange);
	queue.finish();
	queue.enqueueReadBuffer(schlierenbuffer, CL_TRUE, 0, sizeof(iter_t) * s.Resolution * s.Resolution, s.data);
}


void scaledown(iter_t* schlieren_old, iter_t* schlieren_new, int oldres)
{
	int newres = oldres / 2;
	cl::Buffer oldbuffer(context, CL_MEM_READ_WRITE, sizeof(iter_t) * oldres * oldres);
	cl::Buffer newbuffer(context, CL_MEM_READ_WRITE, sizeof(iter_t) * newres * newres);

	cl::Kernel kernel = cl::Kernel(program, "scaledown");

	kernel.setArg(0, oldbuffer);
	kernel.setArg(1, newbuffer);
	kernel.setArg(2, oldres);

	queue.enqueueWriteBuffer(oldbuffer, CL_FALSE, 0, sizeof(iter_t) * oldres * oldres, schlieren_old);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(newres * newres), cl::NullRange/*cl::NDRange(newres)*/);
	queue.finish();
	queue.enqueueReadBuffer(newbuffer, CL_TRUE, 0, sizeof(iter_t) * newres * newres, schlieren_new);
}

int sumup(iter_t* schlieren, int res)
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

int cpu_sumup(iter_t* schlieren, int res)
{
	int sum = 0;
	for (int i = 0; i < res; i++)
		for (int j = 0; j < res; j++)
			sum += schlieren[j * res + i];

	return sum;
}

int main(int argc, char* argv[])
{
	string answer, filename;

	try {
		initOpenCL(device, context, program, queue);
	}
	catch (string s) {
		cout << s << endl << "Could not init OpenCL." << endl;
		return EXIT_FAILURE;
	}

#ifdef MANUAL
	cout << "Generate a data file or Extract information? [G/E]: ";
	cin >> answer;
	if (answer == "E") {
		cout << "Data file path for import: ";
		cin >> answer;
		filename = answer;
		ifstream inputfile(filename, ios::binary);
		if (inputfile.fail()) {
			cout << "Could not open inputfile!" << endl;
			return EXIT_FAILURE;
		}
		SchlierenFile schlierenfile(inputfile);
		cout << "Opened SchlierenFile" << endl;
		cout << schlierenfile << endl;
		
		Factors = factorize(schlierenfile.Resolution);
		cout << "Got factors" << endl;
		return EXIT_SUCCESS;
	
	}
	else if (answer == "G") {
		cout << "Data file path for export: ";
		cin >> answer;
		filename = answer;
		ofstream outputfile(filename, ios::binary);

		cout << "Max iteration: ";
		cin >> MaxIteration;

		if (MaxIteration == 0) {
			cout << "May not be equal to zero!" << endl;
			return EXIT_FAILURE;
		}
		
		unsigned factorcount = 0;
	
		cout << "Number of factors: ";
		cin >> factorcount;

		if (factorcount == 0) {
			cout << "May not be equal to zero!" << endl;
			return EXIT_FAILURE;
		}
		uint32_t factor;
		for (int i = 0; i < factorcount; i++) {
			cout << "Factor #" << i << ": ";
			cin >> factor;
			Factors.push_back(factor);
		}
		Resolution = product(Factors);
						
		SchlierenFile schlierenfile(Resolution, MaxIteration, Scale, Viewport_x, Viewport_y);

		generate_schlieren(schlierenfile);
		schlierenfile.toPNG("test.png", MaxIteration);

		schlierenfile.write(outputfile);
		outputfile.close();

		cout << "Wrote to output file... " << endl;
	}
	else {
		cout << "Please enter G or E...";
		return EXIT_FAILURE;
	}
#else
	ofstream outputfile("test.sch", ios::binary);
	MaxIteration = 100;
	Resolution = 1024;

	SchlierenFile schlierenfile(Resolution, MaxIteration, Scale, Viewport_x, Viewport_y);

	generate_schlieren(schlierenfile);
	for (int i = 0; i < MaxIteration; i++) {
		schlierenfile.toPNG("out/" + to_string(i) + "test.png", i);
	}
	
	cout << schlierenfile << endl;

	schlierenfile.write(outputfile);
	outputfile.close();

#endif

//	cout << "Starting computation" << endl;
//	//testscaledown();
//
//	auto t0 = std::chrono::high_resolution_clock::now();
//	vector<uint32_t> Result = tiling<TileResolution>(TileCount, Iteration, Scale, Viewport_x, Viewport_y);
//	auto t1 = std::chrono::high_resolution_clock::now();
//
//	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
//
//	cout << "Finished after " << duration << "ms" << endl;
//
//#ifdef CSV_EXPORT
//#ifndef CSV_APPEND
//	outfile << "S;B;k;r;N;log r;log N" << endl;
//#endif // !CSV_APPEND
//
//
//
//	for (int i = 0; i < Result.size(); i++) {
//		outfile << Scale << ";";
//		outfile << (TileResolution >> i) * TileCount << ";";
//		outfile << Iteration << ";";
//		outfile << (TileResolution >> i) * TileCount / Scale << ";";
//		outfile << Result[i] << ";";
//		outfile << log10((TileResolution >> i)* TileCount / Scale) << ";";
//		outfile << log10(Result[i]);
//		outfile << endl;
//
//	}

	system("PAUSE");
	return 0;
}

