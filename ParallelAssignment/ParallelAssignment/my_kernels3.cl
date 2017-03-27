﻿///
/// This kernel finds the minimum value in the input vector A, by using a reduction pattern 
/// compare two values from the vector and saving the smallest one. This repeats until the total smallest value 
/// is stored at the first index position of the returned vector B. 
///
__kernel void find_min_val(__global const int* A, __global int* B, __local int* scratch) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//printf("[Min] Value of A = %i\n",A[id]);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] = (scratch[lid] < scratch[lid + i]) ? scratch[lid] : scratch[lid + i];
			//printf("[Min] Comparing: %i  and   %i\n", scratch[lid], scratch[lid + i] );
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) 
	{
		atomic_min(&B[0],scratch[lid]);
	}
}

///
///	This kernel works exactly the same as the min value finder, except with the < symbol reversed
///
__kernel void find_max_val(__global const int* A, __global int* B, __local int* scratch) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//printf("[Max] Value of A = %i\n",A[id]);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] = (scratch[lid] > scratch[lid + i]) ? scratch[lid] : scratch[lid + i];
			//printf("[Max] Comparing: %i  and   %i\n", scratch[lid], scratch[lid + i] );
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) 
	{
		atomic_max(&B[0],scratch[lid]);
	}
}

__kernel void find_mean_val(__global const int* A, __global int* B, __local int* scratch) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//printf("[Mean] Value of A = %i\n",A[id]);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];
			//printf("[Mean] Comparing: %i  and   %i\n", scratch[lid], scratch[lid + i] );
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) 
	{
		atomic_add(&B[0],scratch[lid]);
	}
}

__kernel void find_variance(__global const int* A, __global int* B, int mean) 
{
	int id = get_global_id(0);

	B[id] = A[id] - mean;

	barrier(CLK_LOCAL_MEM_FENCE);

	B[id] = (B[id] * B[id]) / 100;
}

// How to do standard deviation

// Create kernel to find the variance, this will need to take in an additional parameter of the mean of the dataset. 
// 1. Take in vector A
// 2. Minus the mean from each value in A
// 3. barrier break
// 4. Square each value and return as output B.

// These returned values should be summed in the summation kernel then returned
// This sum of the squared values should be divided by N to get the mean of this.
// Sqrt this mean value should get the standard deviation - I think