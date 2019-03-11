# `GPUGA`
Graphics Processing Units Genetic Algorithm

## What is `GPUGA`?
* `GPUGA` stands for Graphics Processing Units Genetic Algorithm. It is a code for empirical potential fitting using the genetic algorithm (GA).
* The first uploaded version is a working code which can optimize a simple testing function. I aim to add features little by little.

## Prerequisites
* You need to have a GPU card with compute capability no less than 3.5 and a `CUDA` toolkit which supports your GPU card installed.
* I have only tested the code in linux operating system.

## Compile GPUGA
* go to the `src` directory and type `make`. When the compilation finishes, an executable named `gpuga` will be generated in the `src` directory. 

## Run GPUGA
* Go to the directory where you can see `src`.
* Type `src/gpuga < examples/input.txt` to run the examples in `examples/`.
* Go to the example directories and run the prepared `Matlab` script to analyze the resutls.

## Author:
* Zheyong Fan (Bohai University and Aalto University)
  * brucenju(at)gmail.com
  * zheyong.fan(at)aalto.fi
  * zheyongfan(at)163.com
