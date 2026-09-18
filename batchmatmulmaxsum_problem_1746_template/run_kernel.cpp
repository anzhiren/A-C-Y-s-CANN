extern "C" void run_kernel(void* kernel_func, void* args, int num_args) {
	// Cast the kernel function pointer to a callable type
	using KernelFuncType = void(*)(void*);
	KernelFuncType kernel = reinterpret_cast<KernelFuncType>(kernel_func);
	// Call the kernel function with the provided arguments
	kernel(args);
}