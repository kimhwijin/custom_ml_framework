//
// Created by 김휘진 on 2022/03/08.
//

#include "LoadCsvDataset.h"
#include <map>
#include <string>
#include <vector>

using namespace std;

int abaloneExecute(){
    std::map<int, std::vector<std::string>> dataset = loadCsvDataset();
    std::cout << dataset[0][0];
}
