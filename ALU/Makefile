CUDA_ROOT = /software/cuda-10.1
ARCH_CC = 70

NVCC = $(CUDA_ROOT)/bin/nvcc
ARCH = arch=compute_$(ARCH_CC),code=sm_$(ARCH_CC)
CUDALIBS = -L$(CUDA_ROOT)/lib64 -lcuda -lcudart
OPT := -Xptxas -O3
NONOPT := -Xptxas -O0

SRC := pipeline.cu
EXE := pipeline

nonOpt: $(SRC)
	@echo "Compiling Non-Optimized Version"
	@$(NVCC) --keep -gencode $(ARCH) $(SRC) $(NONOPT) $(CUDALIBS) -o $(EXE)
	@$(NVCC) -dryrun -gencode $(ARCH) $(SRC) $(NONOPT) $(CUDALIBS) --keep -o $(EXE) 2>dryrun.out 
	@rm pipeline.ptx
	@cp pipeline_temp.ptx pipeline.ptx
	@sed -i "s/^..//g" dryrun.out
	@tail -n 9 dryrun.out >dryrun.out.temp 
	@mv dryrun.out.temp dryrun.out
	@chmod +x dryrun.out
	@./dryrun.out
	@rm dryrun.out

opt: $(SRC)
	@echo "Compiling Optimized Version"
	@$(NVCC) --keep -gencode $(ARCH) $(SRC) $(CUDALIBS) -o $(EXE)
	@$(NVCC) -dryrun -gencode $(ARCH) $(SRC) $(CUDALIBS) --keep -o $(EXE) 2>dryrun.out 
	@rm pipeline.ptx
	@cp pipeline_temp.ptx pipeline.ptx
	@sed -i "s/^..//g" dryrun.out
	@tail -n 9 dryrun.out >dryrun.out.temp 
	@mv dryrun.out.temp dryrun.out
	@chmod +x dryrun.out
	@./dryrun.out
	@rm dryrun.out

clean:
	rm -f *.cpp* *.cudafe* *.fatbin* *.cubin *.module_id *.c
	rm -f pipeline pipeline.ptx *.o
	rm -rf output/*