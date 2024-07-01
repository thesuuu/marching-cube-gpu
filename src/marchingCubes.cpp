#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_runtime.h>
#include <chrono>


#include "utilities.h"
#include "defines.h"


extern "C" void launch_classifyVoxel(dim3 grid, dim3 threads, uint * voxelVerts,
    uint * voxelOccupied, uchar * volume,
    uint3 gridSize, uint3 gridSizeShift,
    uint3 gridSizeMask, uint numVoxels,
    float3 voxelSize, float isoValue);

extern "C" void launch_compactVoxels(dim3 grid, dim3 threads,
    uint * compactedVoxelArray,
    uint * voxelOccupied,
    uint * voxelOccupiedScan, uint numVoxels);

extern "C" void launch_generateTriangles2(
    dim3 grid, dim3 threads, float4 * pos, float4 * norm,
    uint * compactedVoxelArray, uint * numVertsScanned, uchar * volume,
    uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 voxelSize,
    float isoValue, uint activeVoxels, uint maxVerts);

extern "C" void allocateTextures(uint * *d_edgeTable, uint * *d_triTable,
    uint * *d_numVertsTable);
extern "C" void createVolumeTexture(uchar * d_volume, size_t buffSize);
extern "C" void destroyAllTextureObjects();
extern "C" void ThrustScanWrapper(unsigned int* output, unsigned int* input,
    unsigned int numElements);

#if USE_BUCKY
    const char* file_in = "D:/Data/Bucky.raw";
    uint3 gridSizeLog2 = make_uint3(5, 5, 5);
    float isoValue = 0.2f;
#else
    const char* file_in = "D:/Data/sphere_128_uchar.raw";
    uint3 gridSizeLog2 = make_uint3(7, 7, 7);
    float isoValue = 0.2f;
#endif

uint3 gridSizeShift;
uint3 gridSize;
uint3 gridSizeMask;

float3 voxelSize;
uint numVoxels = 0;
uint maxVerts = 0;
uint activeVoxels = 0;
uint totalVerts = 0;

float4* d_pos = 0, * d_normal = 0;

uchar* d_volume = 0;
uint* d_voxelVerts = 0;
uint* d_voxelVertsScan = 0;
uint* d_voxelOccupied = 0;
uint* d_voxelOccupiedScan = 0;
uint* d_compVoxelArray;

// tables
uint* d_numVertsTable = 0;
uint* d_edgeTable = 0;
uint* d_triTable = 0;

uchar* loadRawFile(const char* filename, int size) {
    FILE* fp = fopen(filename, "rb");

    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    uchar* data = (uchar*)malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    printf("Read '%s', %d bytes\n", filename, (int)read);

    return data;
}



template <class T>
void dumpBuffer(T* d_buffer, int nelements, int size_element) {
    uint bytes = nelements * size_element;
    T* h_buffer = (T*)malloc(bytes);
    cudaMemcpy(h_buffer, d_buffer, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nelements; i++) {
        printf("%d: %u\n", i, h_buffer[i]);
    }

    printf("\n");
    free(h_buffer);
}

void cleanup() {

    destroyAllTextureObjects();
    cudaFree(d_edgeTable);
    cudaFree(d_triTable);
    cudaFree(d_numVertsTable);

    cudaFree(d_voxelVerts);
    cudaFree(d_voxelVertsScan);
    cudaFree(d_voxelOccupied);
    cudaFree(d_voxelOccupiedScan);
    cudaFree(d_compVoxelArray);

    if (d_volume) {
        cudaFree(d_volume);
    }
}

void initMC() {
	printf("Initializing Marching Cubes...\n");

    gridSize =
        make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
    gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);
    gridSizeShift =
        make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);

    numVoxels = gridSize.x * gridSize.y * gridSize.z;
    voxelSize =
        make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
    
#if USE_BUCKY
    maxVerts = gridSize.x * gridSize.y * 100;
#else
    maxVerts = gridSize.x * gridSize.y * gridSize.z * 2;
#endif

    printf("grid: %d x %d x %d = %d voxels\n", gridSize.x, gridSize.y, gridSize.z,
        numVoxels);
    printf("max verts = %d\n", maxVerts);

#if SAMPLE_VOLUME
    int size = gridSize.x * gridSize.y * gridSize.z * sizeof(uchar);
    uchar* volume = loadRawFile(file_in, size);

    cudaMalloc((void**)&d_volume, size);
    cudaMemcpy(d_volume, volume, size, cudaMemcpyHostToDevice);
    free(volume);

    createVolumeTexture(d_volume, size);
#endif

    cudaMalloc((void**)&(d_pos), maxVerts * sizeof(float) * 4);
    //cudaMalloc((void**)&(d_normal), maxVerts * sizeof(float) * 4);

    // allocate textures
    allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

    // allocate device memory
    unsigned int memSize = sizeof(uint) * numVoxels;
    cudaMalloc((void**)&d_voxelVerts, memSize);
    cudaMalloc((void**)&d_voxelVertsScan, memSize);
    cudaMalloc((void**)&d_voxelOccupied, memSize);
    cudaMalloc((void**)&d_voxelOccupiedScan, memSize);
    cudaMalloc((void**)&d_compVoxelArray, memSize);

}

void computeIsosurface() {
    int threads = 128;  
    dim3 grid(numVoxels / threads, 1, 1);

    // get around maximum grid size of 65535 in each dimension
    if (grid.x > 65535) {
        grid.y = grid.x / 32768;
        grid.x = 32768;
    }

    // calculate number of vertices need per voxel
    launch_classifyVoxel(grid, threads, d_voxelVerts, d_voxelOccupied, d_volume,
        gridSize, gridSizeShift, gridSizeMask, numVoxels,
        voxelSize, isoValue);
#if DEBUG_BUFFERS
    printf("voxelVerts:\n");
    dumpBuffer(d_voxelVerts, numVoxels, sizeof(uint));
#endif

#if SKIP_EMPTY_VOXELS
    // scan voxel occupied array
    ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

#if DEBUG_BUFFERS
    printf("voxelOccupiedScan:\n");
    dumpBuffer(d_voxelOccupiedScan, numVoxels, sizeof(uint));
#endif

    // read back values to calculate total number of non-empty voxels
    // since we are using an exclusive scan, the total is the last value of
    // the scan result plus the last value in the input array
    {
        uint lastElement, lastScanElement;
        cudaMemcpy((void*)&lastElement,
            (void*)(d_voxelOccupied + numVoxels - 1),
            sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy((void*)&lastScanElement,
            (void*)(d_voxelOccupiedScan + numVoxels - 1),
            sizeof(uint), cudaMemcpyDeviceToHost);
        activeVoxels = lastElement + lastScanElement;
    }

    if (activeVoxels == 0) {
        // return if there are no full voxels
        totalVerts = 0;
        return;
    }

    // compact voxel index array
    launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied,
        d_voxelOccupiedScan, numVoxels);
    //getLastCudaError("compactVoxels failed");

#endif  // SKIP_EMPTY_VOXELS

    // scan voxel vertex count array
    ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);

#if DEBUG_BUFFERS
    printf("voxelVertsScan:\n");
    dumpBuffer(d_voxelVertsScan, numVoxels, sizeof(uint));
#endif

    // readback total number of vertices
    {
        uint lastElement, lastScanElement;
        cudaMemcpy((void*)&lastElement,
            (void*)(d_voxelVerts + numVoxels - 1),
            sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy((void*)&lastScanElement,
            (void*)(d_voxelVertsScan + numVoxels - 1),
            sizeof(uint), cudaMemcpyDeviceToHost);
        totalVerts = lastElement + lastScanElement;
    }

#if SKIP_EMPTY_VOXELS
    dim3 grid2((int)ceil(activeVoxels / (float)NTHREADS), 1, 1);
#else
    dim3 grid2((int)ceil(numVoxels / (float)NTHREADS), 1, 1);
#endif

    while (grid2.x > 65535) {
        grid2.x /= 2;
        grid2.y *= 2;
    }

#if SAMPLE_VOLUME
    launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal, d_compVoxelArray,
        d_voxelVertsScan, d_volume, gridSize, gridSizeShift,
        gridSizeMask, voxelSize, isoValue, activeVoxels,
        maxVerts);
#else
    launch_generateTriangles(grid2, NTHREADS, d_pos, d_normal, d_compVoxelArray,
        d_voxelVertsScan, gridSize, gridSizeShift,
        gridSizeMask, voxelSize, isoValue, activeVoxels,
        maxVerts);
#endif

}


void runMarchingCube() {
	printf("Running Marching Cubes...\n");

    auto start = std::chrono::high_resolution_clock::now();

    // Initialize CUDA buffers for Marching Cubes
    initMC();

    computeIsosurface();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Marching Cubes execution time: " << duration.count() << " ms" << std::endl;
}




int main()
{
    printf("Starting...\n");

    runMarchingCube();
    

    printf("Writing to ply...\n");

    std::vector<float4> vertices(maxVerts);
    cudaMemcpy(vertices.data(), d_pos, maxVerts * sizeof(float4), cudaMemcpyDeviceToHost);
    
    writePLY(vertices, "output.ply");

    cleanup();

    exit(EXIT_SUCCESS);

    //return 0;
}