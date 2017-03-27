#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

///
/// Function takes the filepath and uses ifstream to read in the contents of that file.
/// The contents are read as a string but then the final column of data (after the 5th space character) is parsed to a float
/// The float is * by 100 and saved as an int so it can be passed into OpenCL kernels and still retain the decimal place data
///
vector<int>* readFile(std::string filename)
{
	vector<int>* data = new vector<int>;
	ifstream file (filename);
	string string;
	int spaceCount = 0;

	while (std::getline(file, string))
	{
		std::string tempString;
		for (int i = 0; i < string.size(); i++)
		{
			if (spaceCount < 5)
			{
				if (string[i] == ' ')
				{
					spaceCount++;
				}
			}	
			else
			{
				tempString += string[i];
			}
		}
		data->push_back(std::stof(tempString) * 100);
		spaceCount = 0;
	}
	return data;
}

void print_help() 
{
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv)
{
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	// C:/Users/Computing/Documents/GitHub/ParallelAssignment/ParallelAssignment/x64/Debug/
	vector<int>* data = readFile("C:/Users/Computing/Documents/GitHub/ParallelAssignment/ParallelAssignment/x64/Debug/temp_lincolnshire.txt");
	std::cout << "Reading file complete" << std::endl;

	//detect any potential exceptions
	try 
	{
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try 
		{
			program.build();
		}
		catch (const cl::Error& err) 
		{
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		//Part 4 - memory allocation
		//host - input
		std::vector<mytype> A = {10, 2, 3, 4, 5, 6, 7, 8, 9 , 35};//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 10;

		size_t padding_size = data->size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) 
		{
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			data->insert(data->end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = data->size();//number of input elements
		size_t input_size = data->size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		// Host - Output
		std::vector<mytype> B(input_elements);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes
		
		std::vector<mytype> C(input_elements);
		std::vector<mytype> D(input_elements);
		std::vector<mytype> E(input_elements);


		// Device - Buffers  |  One input buffer and several output buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);

		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, output_size);


		// Copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &(*data)[0]);

		// Zero buffer on device memory for output
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_E, 0, 0, output_size);

		// Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "find_min_val");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		cl::Kernel kernel_2 = cl::Kernel(program, "find_max_val");
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_C);
		kernel_2.setArg(2, cl::Local(local_size * sizeof(mytype)));

		cl::Kernel kernel_3 = cl::Kernel(program, "find_mean_val");
		kernel_3.setArg(0, buffer_A);
		kernel_3.setArg(1, buffer_D);
		kernel_3.setArg(2, cl::Local(local_size * sizeof(mytype)));


		// Call all the kernels in sequence - Except for the mean and standard deviation which require results from these to work
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));


		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_size, &D[0]);

		// Save the results for easier reuse
		float minVal = (float)B[0] / 100.0f;
		float maxVal = (float)C[0] / 100.0f;
		float mean = ((float)D[0] / data->size()) / 100.0f;

		// Create and call the find variance kernel now that the mean is known
		cl::Kernel kernel_4 = cl::Kernel(program, "find_variance");
		kernel_4.setArg(0, buffer_A);
		kernel_4.setArg(1, buffer_E);
		kernel_4.setArg(2, (int)mean);
		queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, output_size, &E[0]);

		float variance = (float)E[0] / 100.0f;

		// ================================== Printing results ================================== //
		std::cout << "\n\n##=================== Results ===================##\n" << std::endl;
		std::cout << "Min = " << minVal << std::endl;
		std::cout << "Max = " << maxVal << std::endl;
		std::cout << "Mean = " << mean << std::endl;
		std::cout << "Variance = " << variance << std::endl;

		system("pause");

	}
	catch (cl::Error err) 
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
