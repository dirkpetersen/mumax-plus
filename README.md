# mumax⁺
A more versatile and extensible GPU-accelerated micromagnetic simulator written in C++ and CUDA with a Python interface. This project is in development alongside its popular predecessor [mumax³](https://github.com/mumax/3).
If you have any questions, feel free to use the [mumax mailing list](https://groups.google.com/g/mumax2).

## Paper

mumax⁺ is described in the following paper:
> mumax+: extensible GPU-accelerated micromagnetics and beyond

https://arxiv.org/abs/2411.18194

Please cite this paper if you would like to cite mumax⁺.

## Dependencies
You should install these yourself
* CUDA Toolkit 10.0 or later
* A C++ compiler which supports C++17, such as GCC
* On Windows (good luck): MSVC 2019

These will be installed automatically within the conda environment
* cmake 4.0.0
* Python 3.13
* pybind11 v2.13.6
* NumPy
* matplotlib
* SciPy
* Sphinx

## Installation from Source

### Linux

Make sure that the following applications and build tools are installed:
* C++ compiler which supports c++17, such as GCC
* CPython *(version 3.8 recommended)* and pip 
* CUDA Toolkit *(version 10.0 or later)*
* git
* miniconda or anaconda

Make especially sure that everything CUDA-related (like `nvcc`) can be found inside your path. This can be done by editing your `~/.bashrc` file and adding the following lines.
```bash
# add CUDA
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```
The paths might differ if CUDA Toolkit has been installed in a different location. If successful, a command such as `nvcc --version` should work.

Clone the mumax⁺ git repository. The `--recursive` flag is used here to get the pybind11 submodule which is needed to build mumax⁺.
```bash
git clone --recursive https://github.com/mumax/plus.git mumaxplus && cd mumaxplus
```
We recommend to install mumax⁺ in a clean conda environment. You could also skip this step and use your own conda environment instead if preferred.
```bash
conda env create -f environment.yml
conda activate mumaxplus
```
Then build and install mumax⁺ using pip.
```bash
pip install -ve .
```
If changes are made to the C++ code, then `pip install -ve .` can be used to rebuild mumax⁺.

You could also compile the source code with double precision, by changing `FP_PRECISION` in `CMakeLists.txt` from `SINGLE` to `DOUBLE` before rebuilding.
```cmake
add_definitions(-DFP_PRECISION=DOUBLE) # FP_PRECISION should be SINGLE or DOUBLE
```

### Windows

**These instructions are old and worked at some point (2021), but not today. If you are brave enough to try Windows and you manage to get it working, please let us know!**

1. Install Visual Studio 2019 and the desktop development with C++ workload
2. Install CUDA Toolkit 10.x
3. Install cmake
4. Download the pybind11 submodule with git
```
git submodule init
git submodule update
```
5. Install Python packages using conda
```
conda env create -f environment.yml
```
6. Build `mumaxplus` using `setuptools`
```
activate mumaxplus
python setup.py develop
```
or `conda`
```
conda activate mumaxplus
conda develop -b .
```

## Building the documentation

Documentation for mumax⁺ follows the [NumPy style guide](https://numpydoc.readthedocs.io/en/latest/format.html) and can be generated using [Sphinx](https://www.sphinx-doc.org). Run the following command in the `docs/` directory to let Sphinx build the HTML documentation pages:
```bash
make html
```
The documentation can now be found at `docs/_build/html/index.html`.

## Multi-GPU Support

mumax⁺ includes support for multi-GPU acceleration using CUDA's cuFFT library for stray field calculations. This feature automatically detects available GPUs and can provide significant performance improvements for large simulations.

### Requirements
- Multiple CUDA-compatible GPUs
- CUDA Toolkit with cuFFT library
- GPUs with peer-to-peer access support (recommended)

### Usage
Multi-GPU support is enabled automatically when multiple GPUs are detected. The system will:
- Automatically detect available GPUs
- Enable peer-to-peer memory access between GPUs
- Use CUDA managed memory for efficient multi-GPU operations
- Fall back gracefully to single-GPU mode if multi-GPU setup fails

### Benchmarking Multi-GPU Performance
Use the provided benchmark script to test multi-GPU performance:
```bash
python examples/multi_gpu_fft_test.py
```

This script compares single-GPU vs multi-GPU performance across different grid sizes and provides detailed timing information.

### Performance Considerations
- Multi-GPU acceleration is most effective for large simulations (grid sizes > 128³)
- Performance gains depend on GPU memory bandwidth and peer-to-peer connectivity
- Optimal performance requires GPUs of similar compute capability
- Memory usage increases due to CUDA managed memory allocation

### Troubleshooting Multi-GPU Issues
If multi-GPU mode fails to initialize, the system automatically falls back to single-GPU mode. Common issues include:
- **Insufficient GPU memory**: Reduce simulation size or use fewer GPUs
- **Driver compatibility**: Ensure CUDA drivers support multi-GPU operations
- **Peer access limitations**: Some GPU configurations don't support peer-to-peer access
- **Mixed GPU architectures**: Performance may be limited by the slowest GPU

Check the console output for detailed error messages and fallback notifications.

## Examples

Lots of example codes are located in the `examples/` directory. They are either simple Python scripts, which can be executed inside said directory like any Python script
```bash
python standardproblem4.py
```
or they are interactive notebooks (`.ipynb` files), which can be run using Jupyter.

### Multi-GPU Examples
- `multi_gpu_fft_test.py` - Benchmark multi-GPU FFT performance for stray field calculations
- `managed_memory_test.py` - Demonstrate CUDA managed memory usage across multiple GPUs

## Testing

Several automated tests are located inside the `test/` directory. Type `pytest` inside the terminal to run them. Some are marked as `slow`, such as `test_mumax3_standardproblem5.py`. You can deselect those by running `pytest -m "not slow"`. Tests inside the `test/mumax3/` directory require external installation of mumax³. They are marked by `mumax3` and can be deselected in the same way.


## Contributing
Contributions are gratefully accepted. To contribute code, fork our repo on GitHub and send a pull request.
