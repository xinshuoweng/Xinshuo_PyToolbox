#ifndef _CAMERA_IO_H_
#define _CAMERA_IO_H_

#include <myheader.h>

//#include "debug_print.h"

class mycamera;
//class ColorCameraModel;

// the following functions should be compatible with Mugsy and Sociopticon

bool LoadIdCalibration(const std::string& calib_fpath, std::vector<mycamera>& out_cameras, const bool consider_skew = false, const bool consider_dist = true);


// deprecated
//bool LoadIdCalibration(const std::string& calib_fpath, std::map<std::string, mycamera>& out_cameras, const bool consider_skew = false, const bool consider_dist = true);


#endif  // _ID_DATA_PARSER_H_
