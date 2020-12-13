/*
** Author(s)      :  Yehia Arafa (yarafa@nmsu.edu)
** 
** File           :  device_kernels.cu  
** 
** Description    :  device kernels declarations
** 
** Paper          :  Y. Arafa et al., "Low Overhead Instruction Latency Characterization
**                                     for NVIDIA GPGPUs," HPEC'19                                                  
*/


/* Miscellaneous & Synchronization Instruction */
__global__ void Ovhd(int *c) {}
__global__ void Nop(int *c) {}
__global__ void BarSync(int *c) {}

/* Movement Instructions */
__global__ void MovSpec(int *c) {}
__global__ void Mov(int *c) {}

/* Conversion Instructions */
__global__ void Cvt(int *c) {}

/* Integer Instructions */
__global__ void Add(int *c) {}
__global__ void Mul(int *c) {}
__global__ void Div(int *c) {} //PTX only [SASS emulated]
__global__ void DivU(int *c) {} //PTX only [SASS emulated]
__global__ void Rem(int *c) {} //PTX only [SASS emulated]
__global__ void RemU(int *c) {} //PTX only [SASS emulated]
__global__ void Mul24Lo(int *c) {}
__global__ void Mul24Hi(int *c) {}
__global__ void Popc(int *c) {}
__global__ void Sad(int *c) {}
__global__ void Clz(int *c) {}
__global__ void Bfind(int *c) {}
__global__ void Brev(int *c) {}
__global__ void Bfe(int *c) {} //PTX only [SASS emulated]
    //--Logic Instructions--//
__global__ void And(int *c) {} //=copysign PTX instruction
__global__ void Cnot(int *c) {} 
    //--Multi Percision Instructions--//
__global__ void MAddc(int *c) {}
    //--Comparison and Selection Instructions--//
__global__ void Setp(int *c) {}

/* Floating Point Instructions */
    //--FP32 Single Precision Instructions--//
__global__ void FAdd(int *c) {}
__global__ void FMul(int *c) {}
__global__ void FFMa(int *c) {}
__global__ void FDiv(int *c) {} //PTX only [SASS emulated]
    //--FP64 Double Precision Instructions--//
__global__ void DFAdd(int *c) {}
__global__ void DFMul(int *c) {}
__global__ void DFFMA(int *c) {}
__global__ void DFDiv(int *c) {} //PTX only [SASS emulated]
    //--FP16 Half Precision Instructions--//
__global__ void HFAdd(int *c) {}
__global__ void HFMul(int *c) {}
__global__ void HFFMa(int *c) {}

/* SFU Special Instructions */
__global__ void Rcp(int *c) {} //PTX only [SASS emulated]
__global__ void FastRcp(int *c) {} //PTX Only [SASS emulated]
__global__ void DRcp(int *c) {} //PTX only [SASS emulated]
__global__ void Sqrt(int *c) {} //PTX only [SASS emulated]
__global__ void FastSqrt(int *c) {} //PTX only [SASS emulated]
__global__ void DSqrt(int *c) {} //PTX only [SASS emulated]
__global__ void Rsqrt(int *c) {} //PTX only [SASS emulated]
__global__ void FastDRsqrt(int *c) {} //=FastDRcp in PTX (or MUFU.RCP64H in SASS) instruction 
__global__ void Sin(int *c) {}
__global__ void Ex2(int *c) {} //= Lg2 in PTX (MUFU.LG2 in SASS). PTX only [SASS emulated]