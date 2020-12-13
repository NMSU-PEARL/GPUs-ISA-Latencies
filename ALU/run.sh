#!/bin/bash

mkdir -p output

#default opt (-O3)
for func in Ovhd Nop MovSpec Cvt Add Mul Div DivU Rem RemU Mul24Lo Mul24Hi Popc Sad Clz Bfind Brev Bfe And MAddc FAdd FMul FFMa Cnot FDiv DFAdd DFMul DFFMA DFDiv HFAdd HFMul HFFMa Rcp FastRcp DRcp Sqrt DSqrt FastSqrt Rsqrt FastDRsqrt Sin Ex2 Setp
do 
	./pipeline $func 3 >> output/$func
	echo $func done
done

# #nonOpt (-O0)
# for func in Ovhd Nop MovSpec Cvt Add Mul Div DivU Rem RemU Mul24Lo Mul24Hi Popc Sad Clz Bfind Brev Bfe And MAddc FAdd FMul FFMa Cnot FDiv DFAdd DFMul DFFMA DFDiv HFAdd HFMul HFFMa Rcp FastRcp DRcp Sqrt DSqrt FastSqrt Rsqrt FastDRsqrt Sin Lg2 Ex2 Setp
# do 
# 	./pipeline $func 0 >> output/$func
# 	echo $func done
# done

echo All done