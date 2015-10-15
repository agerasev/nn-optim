#include <cstdio>
#include <cstdlib>
#include <vector>

#define REDUCE
#define TIME

#include <nn/hw/bp/factory.hpp>
#include <la/vec.hpp>

#ifdef TIME
static unsigned long __time = 0;
#endif

int main(int argc, char *argv[])
{
	const int in_size = 1024, out_size = 1024;
	const int in_step = 16, out_step = 16;
#ifdef REDUCE
	const int reduce_factor = 16;
#endif
	
	FactoryHW factory;
	
	KitHW kit(
	      &factory.getSession()->get_context(), 
	      &factory.getProgram()->get_kernel_map(), 
	      &factory.getSession()->get_queue()
	      );
	LayerHW::BufferHW input(in_size, &kit), output(out_size, &kit);
	ConnHW::BufferHW weight(in_size*out_size, &kit), bias(out_size, &kit);
#ifdef REDUCE
	std::vector<ConnHW::BufferHW *> reduce_buffers;
	{
		int width = (in_size - 1)/reduce_factor + 1;
		while(width > 1) {
			// printf("%d ", width);
			reduce_buffers.push_back(new ConnHW::BufferHW(width*out_size, &kit));
			width = (width - 1)/reduce_factor + 1;
		}
		// printf("\n");
	}
#endif
	
	srand(256);
	weight.randomize();
	bias.randomize();
	
	float *in_data = new float[in_size];
	for(int i = 0; i < in_size; ++i) {
		in_data[i] = (float) i/in_size;
	}
	input.write(in_data);
	delete in_data;
	
#ifndef REDUCE
	cl::kernel *naive = factory.getProgram()->get_kernel("transmit");
#else
	cl::kernel *reduce_init = factory.getProgram()->get_kernel("transmit_reduce_init");
	cl::kernel *reduce = factory.getProgram()->get_kernel("transmit_reduce");
	cl::kernel *reduce_fin = factory.getProgram()->get_kernel("transmit_reduce_finalize");
#endif
	
	for(int iy = out_step; iy < out_size; iy+=out_step)
	{
		for(int ix = in_step; ix < in_size; ix+=in_step)
		{
#ifdef TIME
			unsigned long time = 0;
#endif
			output.clear();
#ifdef REDUCE
			int ixr = ((ix - 1)/reduce_factor + 1);
			reduce_init->evaluate(
			      cl::work_range(ixr,iy), ix, ivec2(ixr, iy), input.getBuffer(), 
			      reduce_buffers[0]->getBuffer(), weight.getBuffer()
			      );
			for(int i = 0; i < (int) reduce_buffers.size() - 1; ++i) {
				int ixrn = ((ixr - 1)/reduce_factor + 1);
				reduce->evaluate(
				      cl::work_range(ixrn,iy), ixr, ivec2(ixrn,iy), 
				      reduce_buffers[i]->getBuffer(), reduce_buffers[i + 1]->getBuffer()
				      );
				ixr = ixrn;
			}
			reduce_fin->evaluate(
			      cl::work_range(iy), ixr, iy, 
			      reduce_buffers[reduce_buffers.size() - 1]->getBuffer(), 
			      output.getBuffer(), bias.getBuffer()
			      );
#ifdef TIME
			time = reduce_init->get_time() + reduce->get_time() + reduce_fin->get_time();
			reduce_init->clear_time();
			reduce->clear_time();
			reduce_fin->clear_time();
#endif
#else
			naive->evaluate(
			      cl::work_range(iy), ix, iy, input.getBuffer(), 
			      output.getBuffer(), weight.getBuffer(), bias.getBuffer()
			      );
#ifdef TIME
			time = naive->get_time();
			naive->clear_time();
#endif
#endif
#ifdef TIME
			if(__time == 0 || time/__time < 2)
				__time = time;
			printf(
			      //"%dx%d, g:%ld, l:%ld : %ld us", 
			      //in_size, out_size, 
			      //range.get_global_size()[0], range.get_local_size()[0], 
			      "%ld",
					  __time
			      );
			printf(" ");
#endif
			factory.getSession()->get_queue().flush();
		}
#ifdef TIME
		printf("\n");
#endif
	}
	
#ifndef TIME
	float *data = new float[out_size];
	output.read(data);
	for(int iy = 0; iy < out_size; ++iy) {
		printf("%f ", data[iy]);
	}
	printf("\n");
	delete data;
#endif
	
#ifdef REDUCE
	for(ConnHW::BufferHW *buffer : reduce_buffers) {
		delete buffer;
	}
#endif
	
	return 0;
}
