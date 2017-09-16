// Author: Xinshuo Weng
// Email: xinshuow@andrew.cmu.edu


#ifndef __MYHEADER_H_INCLUDED__
#define __MYHEADER_H_INCLUDED__

#include "opencv2/core/core.hpp" 

// standard c++ library
#include <iomanip>
#include <iostream>
#include <fstream>
#include <algorithm> // for copy
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <unordered_set>


//#include <vector>

#define default_precision		10
#define default_ransac			100
#define EPS_MYSELF				1e-3
#define EPS_SMALL				1e-10
#define SEED					std::time(NULL)
#define DEBUG_MODE				false
#define default_consider_dist	true
#define default_num_inliers		3
#define default_conf	0.0
#define default_x		0.0
#define default_y		0.0
#define default_anchor	false



#define ASSERT_WITH_MSG(cond, msg) do \
{ if (!(cond)) { std::ostringstream str; str << msg; std::cerr << str.str(); std::abort(); } \
} while(0)



#endif