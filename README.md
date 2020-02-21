# `GPUGA`
Graphics Processing Units Genetic Algorithm

## What is `GPUGA`?
* `GPUGA` stands for Graphics Processing Units Genetic Algorithm. It is a code for empirical potential fitting using the genetic algorithm (GA).
* Using my laptop with a GeForce RTX 2070 GPU card, fitting an empirical potential using `GPUGA` only takes **about one minute**!

## Prerequisites
* You need to have a GPU card with compute capability no less than 3.5 and a `CUDA` toolkit which supports your GPU card installed.
* I have only tested the code in linux operating system.

## Compile GPUGA
* Go to the `src` directory and type `make`. When the compilation finishes, an executable named `gpuga` will be generated in the `src` directory. 

## Run GPUGA
* To run the provided example, go to the directory where you can see `src` and type `src/gpuga < examples/input.txt`
* The example corresponds to the case study for diamond silicon in the paper below.

## Citation
If you use `GPUGA` in your published work, we kindly ask you to cite the following paper which describes the central algorithms used in `GPUGA`:
* Zheyong Fan, Yanzhou Wang, Xiaokun Gu, Ping Qian, Yanjing Su, and Tapio Ala-Nissila,
[A minimal Tersoff potential for diamond silicon with improved descriptions of elastic and phonon transport properties](https://doi.org/10.1088/1361-648X/ab5c5f),
J. Phys.: Condens. Matter **32**, 135901 (2020).

## Author:
* Zheyong Fan (Bohai University and Aalto University)
  * brucenju(at)gmail.com
  * zheyong.fan(at)aalto.fi
  * zheyongfan(at)163.com
  
