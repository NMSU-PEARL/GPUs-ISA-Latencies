/*
** Author(s)      :  Yehia Arafa (yarafa@nmsu.edu)
** 
** File           :  pipeline.cu  
** 
** Description    :  Host (CPU) code to call each device (GPU) microbenchmark to compute
**                   the instructions latencies
** 
** Paper          :  Y. Arafa et al., "Low Overhead Instruction Latency Characterization
**                                     for NVIDIA GPGPUs," HPEC'19                                                  
*/

#include <stdio.h>
#include "device_kernels.cu"

 
int main(int argc, const char* argv[]){

    int n = 10;
    /* Host variable Declaration */
    int *c;
    /* Device variable Declaration */
    int  *d_c;
    /* Allocation of Host Variables */
    c = (int *)malloc(n * sizeof(int));
    /* Allocation of Device Variables */ 
    cudaMalloc((void **)&d_c, n * sizeof(int));

    dim3 Dg = dim3(1); 
    dim3 Db = dim3(1);

// if (argc != 4){
//     printf("wrong number of argument\n"); 
//     exit(0);
// }
//====================== Kernel Start =========================
if(strcmp(argv[1],"Ovhd")==0){ 
    int clck = 0;
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/
        // PTX: 2 x mov
        //SASS: 2 x CS2R
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        clck = overhead;
        printf("PTX: 2 x mov.u32 rx clock;\nSASS: 2 x CS2R.32 Rx SR_CLOCKLO;\n--> number of cycles = %d\n",clck);

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        // PTX: 2 x mov
        //SASS: 2 x (CS2R + Mov) 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        clck = overhead;
        printf("PTX: 2 x mov.u32 rx clock;\nSASS: 2 x CS2R.32 Rx SR_CLOCKLO + 2 x mov;\n--> number of cycles = %d\n",clck);
        /*for the SASS clock overhead only:*/
        // Ovhd<<<Dg, Db>>>(d_c);
        // cudaDeviceSynchronize();
        // cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        // int overhead = c[0];
        // Mov<<<Dg, Db>>>(d_c);
        // cudaDeviceSynchronize();
        // cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        // int mov = (c[0] - overhead)/count_inst;
        // clck = overhead-(2*mov);
        // printf("SASS: \"2 x CS2R.32 Rx SR_CLOCKLO;\"\n--> number of cycles = %d\n",clck);
    }

}else if(strcmp(argv[1],"Nop")==0){
    int count_inst = 4;
    int nop = 0;
    if(atoi(argv[2]) == 3){ /*optimize-(opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Nop<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        nop = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        //BRA.CONV + NOP, count_inst to get each SASS inst lat. = 4*2 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Nop<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        nop = (c[0] - overhead)/count_inst;
    }
    printf("PTX: bar.warp.sync;\nSASS: NOP\n--> number of cycles = %d\n",nop);

}else if(strcmp(argv[1],"BarSync")==0){
    // PTX: bar.sync
    //SASS: BAR.SYNC
    int count_inst = 4;
    int bar = 0;
    if(atoi(argv[2]) == 3){ /*optimize-(opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        BarSync<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        bar = (c[0] - overhead)/count_inst;
        printf("%d\n",c[0]);
        printf("PTX: bar.sync;\nSASS: BAR.SYNC\n--> number of cycles = %d\n",bar);

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        //WARPSYNC + BAR.SYNC
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        BarSync<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        bar = (c[0] - overhead)/count_inst;
        printf("PTX: bar.sync;\nSASS: WARPSYNC + BAR.SYNC\n--> number of cycles = %d\n",bar);
    }   

}else if(strcmp(argv[1],"MovSpec")==0){
    int count_inst = 2;
    int movspec = 0;
    // PTX: mov.u32 r1, tid.x
    //SASS: S2R
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        MovSpec<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        movspec = (c[0] - overhead - 0)/count_inst;
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        MovSpec<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        movspec = (c[0] - overhead - (6*mov))/count_inst;
    }
    printf("PTX: mov.u32 r1, tid.x\nSASS: S2R\n--> number of cycles = %d\n",movspec);

}else if(strcmp(argv[1],"Mov")==0){
    // PTX: mov.u32
    //SASS: MOV
    if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        int count_inst = 6;
        int mov = 0;
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mov = (c[0] - overhead)/count_inst;
        printf("PTX: mov.u32\nSASS: MOV\n--> number of cycles = %d\n",mov);
    }

}else if(strcmp(argv[1],"Cvt")==0){
    int count_inst = 3;
    int cvt = 0;
    // PTX: cvt
    //SASS: F2I
    if(atoi(argv[2]) == 3){ /*optimize-(opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Cvt<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        cvt = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Cvt<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        cvt = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: cvt.rzi.s32.f32\nSASS: F2I.TRUNC.NTZ\n--> number of cycles = %d\n",cvt);

}else if(strcmp(argv[1],"Add")==0){
    int count_inst = 3;
    int add = 0;
    // PTX: add.u32
    //SASS: IADD
    if(atoi(argv[2]) == 3){ /*optimize-(opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Add<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Add<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: add.u32\nSASS: IADD\n--> number of cycles = %d\n",add);

}else if(strcmp(argv[1],"Mul")==0){
    int count_inst = 3;
    int mul = 0;
    // PTX: mul.u32
    //SASS: IMul
    if(atoi(argv[2]) == 3){ /*optimize-(opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mul<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead-4)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Mul<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: mul.lo.u32\nSASS: IMUL\n--> number of cycles = %d\n",mul);

}else if(strcmp(argv[1],"Div")==0){
    int count_inst = 2;
    int div = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    Div<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    div = (c[0] - overhead)/count_inst;
    
    printf("PTX: div.s32\nSASS: [emulated]\n--> number of cycles = %d\n",div);

}else if(strcmp(argv[1],"DivU")==0){
    int count_inst = 2;
    int div = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    DivU<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    div = (c[0] - overhead)/count_inst;

    printf("PTX: div.u32\nSASS: [emulated]\n--> number of cycles = %d\n",div);

}else if(strcmp(argv[1],"Rem")==0){
    int count_inst = 2;
    int rem = 0; 
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    Rem<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    rem = (c[0] - overhead)/count_inst;

    printf("PTX: rem.s32\nSASS: [emulated]\n--> number of cycles = %d\n",rem);

}else if(strcmp(argv[1],"RemU")==0){
    int count_inst = 2;
    int rem = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    RemU<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    rem = (c[0] - overhead)/count_inst;
    
    printf("PTX: rem.u32\nSASS: [emulated]\n--> number of cycles = %d\n",rem);

}else if(strcmp(argv[1],"Mul24Lo")==0){
    int count_inst = 2;
    int mul = 0;
    if(atoi(argv[2]) == 3){ /*optimize-(opt -O3)*/ 
        // PTX: mul24.lo.u32
        //SASS: SGXT.U32 + IMAD, count_inst to get each SASS inst lat. = 2*2
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mul24Lo<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead)/count_inst;
        printf("PTX: mul24.lo.u32\nSASS: [emulated] SGXT.U32 + IMAD\n--> number of cycles = %d\n",mul);

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        // PTX: mul24.lo.u32
        //SASS: SHF.R.U32 + SGXT.U32 + IMAD + Mov, count_inst to get each SASS inst lat. = 2*4
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mul24Lo<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead)/count_inst;
        printf("PTX: mul24.lo.u32\nSASS: [emulated] SHF.R.U32 + SGXT.U32 + IMAD + Mov\n--> number of cycles = %d\n",mul);
    }

}else if(strcmp(argv[1],"Mul24Hi")==0){
    int count_inst = 2;
    int mul = 0;
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        // PTX: mul24.hi.u32
        //SASS: SGXT.U32 + IMAD.WIDE + SHF + PRMT, count_inst to get SASS = 2*3
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mul24Hi<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead)/count_inst;
        printf("PTX: mul24.hi.u32\nSASS: [emulated] SGXT.U32 + IMAD.WIDE.U32 + SHF.R.U32.HI + PRMT\n--> number of cycles = %d\n",mul);

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        // PTX: mul24.hi.u32
        //SASS: SGXT.U32 + IMAD.WIDE + SHF + PRMT + Mov
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mul24Hi<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead)/count_inst;
        printf("PTX: mul24.hi.u32\nSASS: [emulated] SGXT.U32 + IMAD.WIDE + SHF.R.U32.HI + LOP3.LUT + Mov\n--> number of cycles = %d\n",mul);
    }

}else if(strcmp(argv[1],"Popc")==0){
    int count_inst = 3;
    int popc = 0;
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        // PTX: popc.b32
        //SASS: POPC
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Popc<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        popc = (c[0] - overhead)/count_inst;
        printf("PTX: popc.b32\nSASS: POPC\n--> number of cycles = %d\n",popc);
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        // PTX: popc.b32
        //SASS: POPC + 6 x LOP3.LUT + 3 x MOV, count_inst to get each SASS inst lat. = 9
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Popc<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        popc = (c[0] - overhead-(3*mov))/count_inst;
        printf("PTX: popc.b32\nSASS: POPC + LOP3.LUT + MOV\n--> number of cycles = %d\n",popc);
    }
    
}else if(strcmp(argv[1],"Sad")==0){
    int count_inst = 3;
    int sad = 0;
    // PTX: sad.b32
    //SASS: VABSDIFF
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Sad<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        sad = (c[0] - overhead)/count_inst;
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Sad<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        sad = (c[0] - overhead-(4*mov))/count_inst;
    }
    printf("PTX: sad.u32\nSASS: VABSDIFF\n--> number of cycles = %d\n",sad);

}else if(strcmp(argv[1],"Clz")==0){
    int count_inst = 4;
    int clz = 0;
    // PTX: clz.b32
    //SASS: FLO.U32 + IADD
    if(atoi(argv[2]) == 3){ /*optimizer (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Clz<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        clz = (c[0] - overhead)/count_inst;
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Clz<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        clz = (c[0] - overhead-(2*mov))/count_inst;
    }
    printf("PTX: clz.b32\nSASS: FLO.U32 + IADD\n--> number of cycles = %d\n",clz);

}else if(strcmp(argv[1],"Bfind")==0){
    int count_inst = 3;
    int bfind = 0;
    // PTX: bfind.u32
    //SASS: FLO.U32
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Bfind<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        bfind = (c[0] - overhead)/count_inst;
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Bfind<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        bfind = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: bfind.u32\nSASS: FLO.U32\n--> number of cycles = %d\n",bfind);

}else if(strcmp(argv[1],"Brev")==0){
    int count_inst = 4;
    int brev = 0;
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/
        // PTX: brev.u32
        //SASS: BREV + SGXT.U32 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Brev<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        brev = (c[0] - overhead)/count_inst;
        printf("PTX: brev.b32 \nSASS: BREV + SGXT.U32\n--> number of cycles = %d\n",brev);
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        // PTX: brev.u32
        //SASS: BREV + SHF.R.U32.HI + SGXT.U32
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Brev<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        brev = (c[0] - overhead-(2*mov))/2;
        printf("PTX: brev.b32 \nSASS: BREV + SHF.R.U32.HI + SGXT.U32\n--> number of cycles = %d\n",brev);
    }
    
}else if(strcmp(argv[1],"Bfe")==0){
    int count_inst = 4;
    int bfe = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    Bfe<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    bfe = (c[0] - overhead)/count_inst;
    printf("PTX: bfe.u32\nSASS: [emulated]\n--> number of cycles = %d\n",bfe);

}else if(strcmp(argv[1],"And")==0){
    int count_inst = 3;
    int annd = 0;
    // PTX: and.u32
    //SASS: LOP3
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        And<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        annd = (c[0] - overhead)/count_inst;
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        And<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        annd = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: and.b32\nSASS: LOP3.LUT\n--> number of cycles = %d\n",annd);

}else if(strcmp(argv[1],"Cnot")==0){ 
    int count_inst = 2;
    int cnot = 0;
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/
        // PTX: cnot.b32
        //SASS: SETP + SEL 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Cnot<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        cnot = (c[0] - overhead)/count_inst;
        printf("PTX: cnot.b32\nSASS: SETP + SEL\n--> number of cycles = %d\n",cnot);
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        // PTX: cnot.b32
        //SASS: SETP + SEL + IADD3 + 6 x mov
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Cnot<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        cnot = (c[0] - overhead-(2*mov))/count_inst;
        printf("PTX: cnot.b32\nSASS: SETP + SEL + IADD3 + 2 x Mov\n--> number of cycles = %d\n",cnot);
    }
    
}else if(strcmp(argv[1],"MAddc")==0){
    int count_inst = 3;
    int add = 0;
    // PTX: addc.u32
    //SASS: IADD3.X
    if(atoi(argv[2]) == 3){ /*optimize-(opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        MAddc<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead)/count_inst;
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        MAddc<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: addc.u32\nSASS: IADD3.X\n--> number of cycles = %d\n",add);

}else if(strcmp(argv[1],"Setp")==0){ //count_inst = 800
    int count_inst = 800;
    int setp = 0;
    if(atoi(argv[2]) == 3){ /*optimize-(opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Setp<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        setp = (c[0] - (overhead*10000))/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/ 
       count_inst = 10000;
        // SASS: ISETP + PLOP3, count_inst to get SASS inst. = 10000*2 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Setp<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        setp = (c[0] - overhead*10000)/count_inst;
    }
    printf("PTX: setp.ne.s32\nSASS: ISETP.NE.AND\n--> number of cycles = %d\n",setp);

}else if(strcmp(argv[1],"FAdd")==0){ 
    int count_inst = 3;
    int add = 0;
    // PTX: add.f32
    //SASS: FADD
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        FAdd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        FAdd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: add.f32\nSASS: FADD\n--> number of cycles = %d\n",add);

}else if(strcmp(argv[1],"FMul")==0){
    int count_inst = 3;
    int mul = 0;
    // PTX: mul.f32
    //SASS: FMUL
    if(atoi(argv[2]) == 3){ /*optimize-(opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        FMul<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        FMul<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: mul.rn.f32\nSASS: FMUL\n--> number of cycles = %d\n",mul);

}else if(strcmp(argv[1],"FFMa")==0){
    int count_inst = 3;
    int fma = 0;
    // PTX: fma.f32
    //SASS: FFMA
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        FFMa<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        fma = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        FFMa<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        fma = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: fma.rn.f32\nSASS: FFMA\n--> number of cycles = %d\n",fma);

}else if(strcmp(argv[1],"FDiv")==0){
    int count_inst = 2;
    int div = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    FDiv<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    div = (c[0] - overhead)/count_inst;
    
    printf("PTX: div.rn.f32\nSASS: [emulated]\n--> number of cycles = %d\n",div);

}else if(strcmp(argv[1],"DFAdd")==0){
    int count_inst = 2;
    int add = 0;
    // PTX: add.rn.f64
    //SASS: DADD
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        DFAdd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        DFAdd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead-(2*mov))/count_inst;
    }
    printf("PTX: add.rn.f64\nSASS: DADD\n--> number of cycles = %d\n",add);

}else if(strcmp(argv[1],"DFMul")==0){
    int count_inst = 2;
    int mul = 0;
    // PTX: mul.rn.f64
    //SASS: DMUL
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        DFMul<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        DFMul<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead-(2*mov))/count_inst;
    }
    printf("PTX: mul.rn.f64\nSASS: DMUL\n--> number of cycles = %d\n",mul);

}else if(strcmp(argv[1],"DFFMA")==0){
    int count_inst = 2;
    int fma = 0;
    // PTX: fma.rn.f64
    //SASS: DFMA
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        DFFMA<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        fma = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        DFFMA<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        fma = (c[0] - overhead-(2*mov))/count_inst;
    }
    printf("PTX: fma.rn.f64\nSASS: DFMA\n--> number of cycles = %d\n",fma);

}else if(strcmp(argv[1],"DFDiv")==0){
    int count_inst = 2;
    int div = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    DFDiv<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    div = (c[0] - overhead)/count_inst;

    printf("PTX: div.rn.f64\nSASS: [emulated]\n--> number of cycles = %d\n",div);

}else if(strcmp(argv[1],"HFAdd")==0){
    int count_inst = 3;
    int add = 0;
    // PTX: add.f16
    //SASS: HADD2
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        HFAdd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        HFAdd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        add = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: add.f16\nSASS: HADD\n--> number of cycles = %d\n",add);

}else if(strcmp(argv[1],"HFMul")==0){
    int count_inst = 3;
    int mul = 0;
    // PTX: mul.f16
    //SASS: HMUL
    if(atoi(argv[2]) == 3){ /*optimize(opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        HFMul<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        HFMul<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        mul = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: mul.rn.f16\nSASS: HMUL\n--> number of cycles = %d\n",mul);

}else if(strcmp(argv[1],"HFFMa")==0){
    int count_inst = 3;
    int fma = 0;
    // PTX: fma.f16
    //SASS: HFMA
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        HFFMa<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        fma = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        HFFMa<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        fma = (c[0] - overhead-(3*mov))/count_inst;
    }
    printf("PTX: fma.rn.f16\nSASS: HFMA\n--> number of cycles = %d\n",fma);

}else if(strcmp(argv[1],"Rcp")==0){
    int count_inst = 3;
    int rcp = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    Rcp<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    rcp = (c[0] - overhead)/count_inst;

    printf("PTX: rcp.rn.f32\nSASS: [multiple insts including MUFU.RCP]\n--> number of cycles = %d\n",rcp);

}else if(strcmp(argv[1],"FastRcp")==0){
    int count_inst = 3;
    int rcp = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    FastRcp<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    rcp = (c[0] - overhead)/count_inst;

    printf("PTX: rcp.approx.f32\nSASS: [multiple insts including MUFU.RCP]\n--> number of cycles = %d\n",rcp);

}else if(strcmp(argv[1],"DRcp")==0){
    int count_inst = 2;
    int rcp = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    DRcp<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    rcp = (c[0] - overhead)/count_inst;

    printf("PTX: rcp.rn.f64\nSASS: [multiple insts including MUFU.RCP64H]\n--> number of cycles = %d\n",rcp);

}else if(strcmp(argv[1],"Sqrt")==0){
    int count_inst = 3;
    int sqrt = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    Sqrt<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    sqrt = (c[0] - overhead)/count_inst;

    printf("PTX: sqrt.rn.f32\nSASS: [multiple insts including MUFU.RSQ]\n--> number of cycles = %d\n",sqrt);

}else if(strcmp(argv[1],"DSqrt")==0){
    int count_inst = 2;
    int sqrt = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    DSqrt<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    sqrt = (c[0] - overhead)/count_inst;

    printf("PTX: sqrt.rn.f64\nSASS: [multiple insts including MUFU.RSQ64]\n--> number of cycles = %d\n",sqrt);

}else if(strcmp(argv[1],"FastSqrt")==0){
    int count_inst = 3;
    int sqrt = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    FastSqrt<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    sqrt = (c[0] - overhead)/count_inst;

    printf("PTX: sqrt.approx.f32\nSASS: [multiple insts including MUFU.SQRT]\n--> number of cycles = %d\n",sqrt);

}else if(strcmp(argv[1],"Rsqrt")==0){
    int count_inst = 3;
    int sqrt = 0;
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    Rsqrt<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    sqrt = (c[0] - overhead)/count_inst;

    printf("PTX: rsqrt.approx.f32\nSASS: [multiple insts including MUFU.RSQ]\n--> number of cycles = %d\n",sqrt);

}else if(strcmp(argv[1],"FastDRsqrt")==0){
    int count_inst = 4;
    int sqrt = 0;
    // PTX: rsqrt.approx.ftz.f64
    //SASS: MUFU.RSQ64H
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        FastDRsqrt<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        sqrt = (c[0] - overhead)/count_inst;

    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        FastDRsqrt<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        sqrt = (c[0] - overhead-(12*mov))/count_inst;
    }
    printf("PTX: rsqrt.approx.ftz.f64\nSASS: MUFU.RSQ64H\n--> number of cycles = %d\n",sqrt);

}else if(strcmp(argv[1],"Sin")==0){
    int count_inst = 1;
    int sin = 0;
    // PTX: sin.approx.f32
    //SASS: FMUL + MUFU.SIN, count_inst to get each SASS inst lat. = 2 * 2
    if(atoi(argv[2]) == 3){ /*optimize (opt -O3)*/ 
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Sin<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        sin = (c[0] - overhead)/count_inst;
    }else if(atoi(argv[2]) == 0){ /*non-optimize (nonOpt -O0)*/
        Ovhd<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int overhead = c[0];
        Mov<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        int mov = (c[0] - overhead)/6;
        Sin<<<Dg, Db>>>(d_c);
        cudaDeviceSynchronize();
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        sin = (c[0] - overhead-(2*mov))/count_inst;
    }
    printf("PTX: sin.approx.f32\nSASS: FMUL + MUFU.SIN\n--> number of cycles = %d\n",sin);

}else if(strcmp(argv[1],"Ex2")==0){
    int count_inst = 3;
    int ex = 0;
    // PTX: ex2.approx.f32
    //SASS: FSTEP + FMUL + MUFU.LG2 + FADD, count_inst to get each SASS inst lat. = 3 * 4
    Ovhd<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    int overhead = c[0];
    Ex2<<<Dg, Db>>>(d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    ex = (c[0] - overhead)/count_inst;
    printf("PTX: ex2.approx.f32\nSASS: FSTEP + FMUL + MUFU.EX2 + FMUL\n--> number of cycles = %d\n",ex);

}else { 
    printf("Wrong Instruction\n"); 
    exit(0);
}

    /* Free Device Memory */
    cudaFree(d_c);
    /* Free Host Memory */
    free(c);
    

    return 0;
}