#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl2.hpp>
#include "SchlierenFormat.h"
#include "lodepng.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
