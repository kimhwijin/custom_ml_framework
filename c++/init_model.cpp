//
// Created by 김휘진 on 2022/03/08.
//
#include "LoadCSVDataset.h"
#include <map>
#include <string>
#include <vector>

using namespace std;

int main(){
    std::map<int, std::vector<string>> dataset = loadCSVDataset();
    std::cout << dataset[0][0];
}
