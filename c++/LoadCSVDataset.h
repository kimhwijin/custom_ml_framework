//
// Created by 김휘진 on 2022/03/08.
//

#ifndef C___LOADCSVDATASET_H
#define C___LOADCSVDATASET_H

#endif //C___LOADCSVDATASET_H
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>

using std::cout; using std::cerr;
using std::endl; using std::string;
using std::ifstream; using std::ostringstream;
using std::istringstream;

string readFileIntoString(const string& path);
std::map<int, std::vector<string>> loadCSVDataset();