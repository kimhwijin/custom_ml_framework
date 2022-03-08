#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>

using std::cout; using std::cerr;
using std::endl; using std::string;
using std::ifstream; using std::ostringstream;
using std::istringstream;

string readFileIntoString(const string& path){
    auto ss = ostringstream{};
    ifstream csvFile(path);
    if (!csvFile.is_open()){
        cerr << "Could not open the file - '" << path << "'" << endl;
        exit(EXIT_FAILURE);
    }
    ss << csvFile.rdbuf();
    return ss.str();
}

std::map<int, std::vector<string>> loadCSVDataset() {
    string filename("/Users/hwijin/Desktop/Code/ml/ml_framework/dataset/abalone.data.csv");
    string file_contents;
    std::map<int, std::vector<string>> csv_contents;

    char delimiter = ',';
    file_contents = readFileIntoString(filename);

    istringstream sstream(file_contents);
    std::vector<string> items;
    string record;

    int counter = 0;
    while (std::getline(sstream, record)){
        istringstream  line(record);
        while (std::getline(line, record, delimiter)){
            items.push_back(record);
        }
        csv_contents[counter] = items;
        items.clear();
        counter += 1;
    }
    return csv_contents;
}
