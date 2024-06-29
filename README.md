# aoc-2019-16

CUDA implementation of a simple brute force algorithm to solve [Advent of Code 2019, day 16, part 2](https://adventofcode.com/2019/day/16).

## Usage
**Please note:** A CUDA capable device must be installed. For example, an NVIDIA Geforce 2000, 3000, or 4000 series GPU.

`aoc-2019-16 <signal file | raw signal>`

The application will first try to open the argument as a file and use its content as the signal. If that fails, the argument itself will be used as the signal instead.