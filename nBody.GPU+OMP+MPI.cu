//REFERENCE : //https://software.intel.com/en-us/forums/intel-c-compiler/topic/560682#comment-1829971
#undef SEEK_SET 
#undef SEEK_CUR
#undef SEEK_END

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


#define BODIES    10000000 //for production
#define TIMESTEPS 1000     //for production
//#define BODIES 50000
//#define TIMESTEPS 3

#define GRAVCONST 0.0000001

// global vars
float mass[BODIES];
float vx[BODIES], vy[BODIES];
float x[BODIES], y[BODIES];
float dx, dy, d, F, ax, ay;

//MPI vars
int numOfRanks, rankID; 	//number of ranks (processes), and rank ID
double t1,t2,t3,t4;			//for timing		
int *numOfBODIESperRank, *offsets;	//array of size of BODIES, and for index/displacements

//GPU vars
float *dev_mass;
float *dev_x;
float *dev_y;
float *dev_vx;
float *dev_vy;
int block, threadsPerBlock;

__global__ void simulationKernel(int start, int end, float *mass, float *x, float *y, float *vx, float *vy);
__global__ void positionKernel(int start, int end, float *x, float *y, float *vx, float *vy);

void randomInit();
void outputBody(int);

int main(int argc, char** argv) {
  int time,i;

  MPI_Init(NULL, NULL);								//initialise MPI parallel environment
  MPI_Comm_size(MPI_COMM_WORLD, &numOfRanks);		//set the number of processes in the communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rankID);			//sets the rankID of the process in the communicator MPI_COMM_WORLD
  
  //begin timing of WHOLE CODE
   t1=MPI_Wtime(); 
   
//dev timer
cudaEvent_t startCuda, stopCuda;
cudaEventCreate(&startCuda); cudaEventCreate(&stopCuda); //create timing checkpoint for CUDA   									//start timing of the WHOLE CODE

//create index for MPI, so that every processes do not overlap each other
offsets=(int *) malloc(sizeof(int)*numOfRanks);
numOfBODIESperRank=(int *) malloc(sizeof(int)*numOfRanks);

if(argc!=2){threadsPerBlock=1024;} else {threadsPerBlock=atoi(argv[1]);} //set the num of threadPerBlock  

if(rankID==0) {
			int remainder = BODIES % numOfRanks;					//calculate the remainder of indivisible amount of the total number of BODIES divided by number of ranks
			int sum = 0;											//accumulation of each of number of BODIES per rank 
			for(i = 0; i < numOfRanks; i++) {						
				numOfBODIESperRank[i] = BODIES / numOfRanks;		//split the amount of BODIES per rank
				if (remainder > 0) {								//if not divisible by number of rank, the remainder will distributed into ranks, start from rankID=0 (root), the next remainder will be given to next rank, etc
					numOfBODIESperRank[i] += 1;
					remainder--;
				}
				offsets[i] = sum;									//displacement/offset/ start of the index to be distributed 
				sum += numOfBODIESperRank[i];
			}
			randomInit();
}

MPI_Bcast(numOfBODIESperRank,numOfRanks,MPI_FLOAT,0,MPI_COMM_WORLD); //broadcast all length and index of the portions of the bodies for each processes
MPI_Bcast(offsets,numOfRanks,MPI_FLOAT,0,MPI_COMM_WORLD);  				//broadcast all length and index of the portions of the bodies for each processes

float scatter_vx[numOfBODIESperRank[rankID]],scatter_vy[numOfBODIESperRank[rankID]]; //allocate 'local' velocities
MPI_Bcast(mass, BODIES, MPI_FLOAT, 0, MPI_COMM_WORLD);	//broadcast all mass of the bodies for each processes
MPI_Bcast(x, BODIES, MPI_FLOAT, 0, MPI_COMM_WORLD);	//broadcast all x-axis of the bodies for each processes
MPI_Bcast(y, BODIES, MPI_FLOAT, 0, MPI_COMM_WORLD);	//broadcast all y-axis of the bodies for each processes
MPI_Scatterv(vx,numOfBODIESperRank,offsets, MPI_FLOAT,scatter_vx,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD); //scatter 'local' velocities
MPI_Scatterv(vy,numOfBODIESperRank,offsets, MPI_FLOAT,scatter_vy,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD); //scatter 'local' velocities

t2=MPI_Wtime(); //start of the timing of WORK REGION

cudaEventRecord(startCuda,0);  //start of CUDA WORK REGION

    for (time=0; time<TIMESTEPS; time++) {
    //printf("Timestep %d\n",time);

//printf("Timestep=%d rankID=%d offsets=%d end=%d\n",time,rankID,offsets[rankID],offsets[rankID]+numOfBODIESperRank[rankID]);

		  //CUDA
		  //memory allocation for device vars
			cudaMalloc(&dev_mass, BODIES*sizeof(float));
			cudaMalloc(&dev_x,BODIES*sizeof(float));
			cudaMalloc(&dev_y, BODIES*sizeof(float));
			cudaMalloc(&dev_vx, numOfBODIESperRank[rankID]*sizeof(float));
			cudaMalloc(&dev_vy, numOfBODIESperRank[rankID]*sizeof(float));
			//copy variable from host to device
			cudaMemcpy(dev_mass, mass, BODIES*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_x, x, BODIES*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_y, y, BODIES*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_vx, scatter_vx, numOfBODIESperRank[rankID]*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_vy, scatter_vy, numOfBODIESperRank[rankID]*sizeof(float), cudaMemcpyHostToDevice);		
			
			////launch kernel, for simulation and integrating the velocities, 
			block = ceil((float) BODIES/(float)threadsPerBlock);
			simulationKernel <<<block, threadsPerBlock>>> (offsets[rankID], offsets[rankID]+numOfBODIESperRank[rankID],dev_mass,dev_x,dev_y,dev_vx,dev_vy);
			positionKernel <<<block, threadsPerBlock>>> (offsets[rankID], offsets[rankID]+numOfBODIESperRank[rankID],dev_x,dev_y,dev_vx,dev_vy);
			//copy variables from device to host
			cudaMemcpy(mass, dev_mass, BODIES*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(x, dev_x, BODIES*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(y, dev_y, BODIES*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(scatter_vx, dev_vx, numOfBODIESperRank[rankID]*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(scatter_vy, dev_vy, numOfBODIESperRank[rankID]*sizeof(float), cudaMemcpyDeviceToHost);
			//print error from device, if any, and clear memory allocation
			cudaError err = cudaGetLastError(); if ( err != cudaSuccess) {printf("CUDA RT error: %s \n",cudaGetErrorString(err));}
			cudaFree(dev_mass);cudaFree(dev_x);cudaFree(dev_y);cudaFree(dev_vx);cudaFree(dev_vy);

    //gather and send the new position of each body to all other ranks
    MPI_Allgatherv(&x[offsets[rankID]], numOfBODIESperRank[rankID], MPI_FLOAT, &x, numOfBODIESperRank, offsets, MPI_FLOAT, MPI_COMM_WORLD);
	MPI_Allgatherv(&y[offsets[rankID]], numOfBODIESperRank[rankID], MPI_FLOAT, &y, numOfBODIESperRank, offsets, MPI_FLOAT, MPI_COMM_WORLD);

    //printf("---\n");
  } // time
  
  cudaEventRecord(stopCuda,0); //end of CUDA WORK REGION

  t3=MPI_Wtime(); //end of the timing of WORK REGION
  
  //gathering local velocities for output
MPI_Gatherv(scatter_vx, numOfBODIESperRank[rankID], MPI_FLOAT, &vx, numOfBODIESperRank, offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv(scatter_vy, numOfBODIESperRank[rankID], MPI_FLOAT, &vy, numOfBODIESperRank, offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if(rankID==0) {
printf("Final data\n");
//worksharing the output in CPU
  #pragma omp parallel for 
  //#pragma omp single	//uncomment this left code and comment above : for ordering output
	  for (i=0; i<BODIES; i++) {
		outputBody(i);
		}
  t4=MPI_Wtime();  //end of the timing of WHOLE CODE

float eTime;
cudaEventElapsedTime(&eTime, startCuda, stopCuda); //calculate time spent on device work

printf("\ttime CPU (WHOLE CODE):%f seconds\tCPU (WORK REGION):%f seconds\tGPU (WORK REGION):%f seconds\tblock(s)=%d\tthread(s)PerBlock=%d\n",t4-t1,t3-t2,(float)eTime/1000,block,threadsPerBlock);   
}
  
  MPI_Finalize();		//end of the MPI
}

void randomInit() {
  int i;
  
	   for (i=0; i<BODIES; i++) {
		mass[i] = 0.001 + (float)rand()/(float)RAND_MAX;            // 0.001 to 1.001

		x[i] = -250.0 + 500.0*(float)rand()/(float)RAND_MAX;   //  -10 to +10 per axis
		y[i] = -250.0 + 500.0*(float)rand()/(float)RAND_MAX;   //

		vx[i] = -0.2 + 0.4*(float)rand()/(float)RAND_MAX;   // -0.25 to +0.25 per axis
		vy[i] = -0.2 + 0.4*(float)rand()/(float)RAND_MAX;   
	   }
  
  printf("Randomly initialised\n");
  return;
}


void outputBody(int i) {
  printf("Body %d: Position=(%f,%f) Velocity=(%f,%f)\n", i, x[i],y[i], vx[i],vy[i]);
  return;
}

__global__ void simulationKernel(int start, int end, float *mass, float *x, float *y, float *vx, float *vy){ 
 // parallel control via varying index
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 //int j=blockIdx.y*blockDim.y+threadIdx.y;
 int j;
float dx, dy, d, F, ax, ay;			//distance in x-axis position, y-axis position, d distance, Force, accelleration in x-axis, velocity in y-axis of a body
 
    if (start<=i && i<end) {
		  // calc forces on body i due to bodies (j != i)
		   for (j=0; j<BODIES; j++)  {
		  //if (j<BODIES)  {
				if (j != i) {
					//calculate the x-axis & y-axis distances (r) body i due to bodies (j != i)
				  dx = x[j] - x[i];
				  dy = y[j] - y[i];
				  //r^2 = dx^2 + dy^2, finally 'bug' has been fixed (compared to assignment 1, where r=dx^2 + dy^2)
				  float temp=sqrt(dx*dx + dy*dy);
				  //catch if the distance is too narrow/small
				  d = temp>0.01 ? temp : 0.01;
				  //calculate the Force by applying Newton's Law
				  F = GRAVCONST * mass[i] * mass[j] / (d*d);
				   //calculate the acceleration of body i in x-axis and y-axis
				  ax = (F/mass[i]) * dx/d;
				  ay = (F/mass[i]) * dy/d;
				  int scatter_i = i - start;
				  //the integral of acceleration (over time) of i body, become its velocity. We calculate velocity in x-axis and y-axis.
				  vx[scatter_i] += ax;
				  vy[scatter_i] += ay;
				}
		  } // body j
    } // body i 
}

// having worked out all velocities we now apply and determine new position
__global__ void positionKernel(int start, int end, float *x, float *y, float *vx, float *vy){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (start<=i && i<end) {
		int scatter_i = i - start;
      x[i] += vx[scatter_i];
      y[i] += vy[scatter_i];
    }
}
