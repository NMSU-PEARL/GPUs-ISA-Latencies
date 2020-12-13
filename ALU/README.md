## ALU Instructions Microbenchmarks

The CPU host file (*pipeline.cu*) launches each GPU instruction's microbenchmark to compute its cycle latency

### Usage

* Configure the path of  Cuda toolkit (*nvcc*) in the Makefile
  * ***It is recommended to use cuda 10.1 as the code haven't been tested with cuda 11 just yet***

* Configure the **ARCH_CC** variable in the Makefile depending on the target NVIDIA GPU architecture.   
  - ***Volta TITAN V*** has a 70 SM arch generation and a 70 compute capabilty. Thus, ARCH_CC =70,  
  - ***Turing TITAN RTX*** has a 75 SM arch generation and a 75 compute capabilty. Thus, ARCH_CC =75

* To compile:

    ```
    make opt 
    ```
 * To run:
    * Either run each kernel in **device_kernel.cu** seperatly or use **run.sh** to compute all the kernels, the results will be saved in the output folder
    * The command to run is as flollows:
      ```
      ./pipeline Add 3
      ```
    Where "Add" is kernel name (see **device_kernels.cu**), "3" for O3, and "0" for O0
