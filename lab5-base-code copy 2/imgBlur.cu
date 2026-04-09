#include "libwb/wb.h"
#include "my_timer.h"

#define wbCheck(stmt)							\
  do {									\
    cudaError_t err = stmt;						\
    if (err != cudaSuccess) {						\
      wbLog(ERROR, "Failed to run stmt ", #stmt);			\
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));	\
      return -1;							\
    }									\
  } while (0)

#define BLUR_SIZE 21

///////////////////////////////////////////////////////
//@@ INSERT YOUR CODE HERE

#define BLOCK_SIZE 16
#define L_PARAM 2
#define OUTPUT_DIM (L_PARAM * BLOCK_SIZE)
#define TILE_DIM (OUTPUT_DIM + 2 * BLUR_SIZE)

__global__ void blurKernel(float *out, float *in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 for warp alignment
    __shared__ float rowSum[TILE_DIM][OUTPUT_DIM + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outBlockX = blockIdx.x * OUTPUT_DIM;
    int outBlockY = blockIdx.y * OUTPUT_DIM;

    // Load shared tile 
    for (int y = ty; y < TILE_DIM; y += BLOCK_SIZE) {
        for (int x = tx; x < TILE_DIM; x += BLOCK_SIZE) {
            int curRow = outBlockY + y - BLUR_SIZE;
            int curCol = outBlockX + x - BLUR_SIZE;

            if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                tile[y][x] = in[curRow * width + curCol];
            else
                tile[y][x] = 0.0f; // Zero padding if outside
        }
    }

    __syncthreads(); 

    // Horizontal partial sums
    for (int y = ty; y < TILE_DIM; y += BLOCK_SIZE) {
        for (int x = tx; x < OUTPUT_DIM; x += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int k = 0; k < 2 * BLUR_SIZE + 1; ++k) {
                sum += tile[y][x + k];
            }
            rowSum[y][x] = sum;
        }
    }

    __syncthreads();

    // Final vertical sum
    for (int y = ty; y < OUTPUT_DIM; y += BLOCK_SIZE) {
        for (int x = tx; x < OUTPUT_DIM; x += BLOCK_SIZE) {
            int row = outBlockY + y;
            int col = outBlockX + x;

            if (row < height && col < width) {
                float sum = 0.0f;
                for (int i = 0; i < 2 * BLUR_SIZE + 1; ++i) {
                    sum += rowSum[y + i][x];
                }

                int y0 = max(0, row - BLUR_SIZE);
                int y1 = min(height - 1, row + BLUR_SIZE);
                int x0 = max(0, col - BLUR_SIZE);
                int x1 = min(width - 1, col + BLUR_SIZE);
                int count = (y1 - y0 + 1) * (x1 - x0 + 1);

                out[row * width + col] = sum / count;
            }
        }
    }
}
///////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  wbImage_t goldImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *goldOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage = wbImport(inputImageFile);

  char *goldImageFile = argv[2];
  goldImage = wbImport(goldImageFile);

  // The input image is in grayscale, so the number of channels is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get host input and output image data
  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  goldOutputImageData = wbImage_getData(goldImage);

  // Start timer
  timespec timer = tic();

  ////////////////////////////////////////////////
  //@@ INSERT AND UPDATE YOUR CODE HERE
  size_t numBytes = imageWidth * imageHeight * sizeof(float);

  wbCheck(cudaMalloc((void **)&deviceInputImageData, numBytes));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numBytes));

  // Fast pinned-to-device DMA transfer setup
  wbCheck(cudaHostRegister(hostInputImageData, numBytes, cudaHostRegisterDefault));
  wbCheck(cudaHostRegister(hostOutputImageData, numBytes, cudaHostRegisterDefault));
  
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, numBytes, cudaMemcpyHostToDevice));
   
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 dimGrid((imageWidth  + OUTPUT_DIM - 1) / OUTPUT_DIM,
               (imageHeight + OUTPUT_DIM - 1) / OUTPUT_DIM, 1);

  for(int i = 0; i < 10; i++) {
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData,
                                      deviceInputImageData, 
                                      imageWidth, imageHeight);
  }

  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, numBytes, cudaMemcpyDeviceToHost));

  wbCheck(cudaHostUnregister(hostInputImageData));
  wbCheck(cudaHostUnregister(hostOutputImageData));
  ///////////////////////////////////////////////////////
  
  // Stop and print timer
  toc(&timer, "GPU execution time (including data transfer) in seconds");

  // Check the correctness of your solution
  //wbSolution(args, outputImage);

  for(int i=0; i<imageHeight; i++){
    for(int j=0; j<imageWidth; j++){
      if(abs(hostOutputImageData[i*imageWidth+j]-goldOutputImageData[i*imageWidth+j])/goldOutputImageData[i*imageWidth+j]>0.01){
        printf("Incorrect output image at pixel (%d, %d): goldOutputImage = %f, hostOutputImage = %f\n", i, j, goldOutputImageData[i*imageWidth+j],hostOutputImageData[i*imageWidth+j]);
  return -1;
      }
    }
  }
  printf("Correct output image!\n");

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}