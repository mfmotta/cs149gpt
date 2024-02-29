
## The C++ API 

- The at::Tensor class in ATen is not differentiable by default. To add the differentiability of tensors the autograd API provides, you must use **tensor factory functions from the torch:: namespace** instead of the at:: namespace. For example, while a tensor created with at::ones will not be differentiable, a tensor created with torch::ones will be.

</br>

## Cpp Extension 

[source](https://pytorch.org/tutorials/advanced/cpp_extension.html)


- <torch/extension.h> is the one-stop header to include all the necessary PyTorch bits to write C++ extensions. It includes:

    - The ATen library, which is our primary API for tensor computation,
    - pybind11, which is how we create Python bindings for our C++ code,
    - Headers that manage the details of interaction between ATen and pybind11.


### [JIT Compiling Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html#jit-compiling-extensions)

The gpt149.py function uses the  (just in time) JIT compiling extension instead of [setuptools](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension) to build C++:

```
from torch.utils.cpp_extension import load
mr = load(name="custom_module", sources=["module.cpp"],  extra_cflags=["-mavx", "-O3", "-fopenmp"], extra_ldflags=[ispc_path])
```

In the background, this will do the following:

    - Create a temporary directory /tmp/torch_extensions/custom_module,
    - Emit a Ninja build file into that temporary directory,
    - Compile your source files into a shared library,
    - Import this shared library as a Python module.

In fact, if you pass verbose=True to cpp_extension.load(), you will be informed about the process:

    ```
    Using /tmp/torch_extensions as PyTorch extensions root...
    Emitting ninja build file /tmp/torch_extensions/custom_module/build.ninja...
    Building extension module custom_module...
    Loading extension module custom_module...
    ```

### [Writing a Mixed C++/CUDA extension](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension)

We can hand-write parts of the forward and backward passes with custom CUDA kernels. 

The general strategy for writing a CUDA extension is to first write a C++ file which defines the functions that will be called from Python, and binds those functions to Python with pybind11. Furthermore, this file will also declare functions that are defined in CUDA (.cu) files. The C++ functions will then do some checks and ultimately forward its calls to the CUDA functions. In the CUDA files, we write our actual CUDA kernels. The cpp_extension package will then take care of compiling the C++ sources with a C++ compiler like gcc and the CUDA sources with NVIDIA’s nvcc compiler. This ensures that each compiler takes care of files it knows best to compile. Ultimately, they will be linked into one shared library that is available to us from Python code.


TODO: 

split module.cpp into a cpp and a cuda file as in this documentation example.

Use shared memory 

A setup could also look like [this](https://pytorch.org/tutorials/advanced/cpp_extension.html#integrating-a-c-cuda-operation-with-pytorch)

See also [this setup example](https://github.com/ClementPinard/Pytorch-Correlation-extension/blob/14a159ebad5399adf5db965bb5cab095c98ebc87/setup.py#L32) -- here CUDA function uses shared memory, see [related discussion](https://github.com/pytorch/extension-cpp/issues/59). 


### AT_DISPATCH_FLOATING_TYPES

Notice the macro AT_DISPATCH_FLOATING_TYPES macro in the documnetation. While ATen abstracts away the device and datatype of the tensors we deal with, a tensor will, at runtime, still be backed by memory of a concrete type on a concrete device. As such, we need a way of **determining at runtime what type a tensor is and then selectively call functions with the corresponding correct type signature**. Done manually, this would (conceptually) look something like this:

switch (tensor.type().scalarType()) {
  case torch::ScalarType::Double:
    return function<double>(tensor.data<double>());
  case torch::ScalarType::Float:
    return function<float>(tensor.data<float>());
  ...
}
The purpose of AT_DISPATCH_FLOATING_TYPES is to take care of this dispatch for us. It takes a type (gates.type() in the docu example case), a name (for error messages) and a lambda function. Inside this lambda function, the type alias scalar_t is available and is defined as the type that the tensor actually is at runtime in that context. As such, if we have a template function (which our CUDA kernel will be), we can instantiate it with this scalar_t alias, and the correct function will be called. In this case, we also want to retrieve the data pointers of the tensors as pointers of that scalar_t type. If you wanted to dispatch over all types and not just floating point types (Float and Double), you can use AT_DISPATCH_ALL_TYPES.

Note that we perform some operations with plain ATen. These operations will still run on the GPU, but using ATen’s default implementations. This makes sense because ATen will use highly optimized routines for things like matrix multiplies (e.g. addmm) or convolutions which would be much harder to implement and improve ourselves.


