#include <fstream>
#include <iostream>
#include <vector>

void writePLY(const std::vector<float4>& vertices, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file: " << filename << std::endl;
        return;
    }

    // 写入 PLY 文件头部
    outFile << "ply\n";
    outFile << "format ascii 1.0\n";
    outFile << "element vertex " << vertices.size() << "\n";
    outFile << "property float x\n";
    outFile << "property float y\n";
    outFile << "property float z\n";
    outFile << "element face " << vertices.size() / 3 << "\n";
    outFile << "property list uchar int vertex_indices\n";
    outFile << "end_header\n";

    // 写入顶点数据
    for (const auto& vertex : vertices) {
        outFile << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
    }

    // 写入面片数据
    for (size_t i = 0; i < vertices.size(); i += 3) {
        outFile << "3 " << i << " " << i + 1 << " " << i + 2 << "\n";
    }

    outFile.close();
}







