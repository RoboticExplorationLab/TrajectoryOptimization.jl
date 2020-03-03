# Benchmark Report for *TrajectoryOptimization*

## Job Properties
* Time of benchmark: 2 Mar 2020 - 17:33
* Package commit: dirty
* Julia commit: 2d5741
* Julia command flags: None
* Environment variables: None

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                           | time            | GC time | memory          | allocations |
|------------------------------|----------------:|--------:|----------------:|------------:|
| `["ALTRO", "3obs"]`          | 960.824 μs (5%) |         | 980.07 KiB (1%) |         883 |
| `["ALTRO", "cartpole"]`      |   4.836 ms (5%) |         |  22.13 KiB (1%) |         508 |
| `["ALTRO", "double_int"]`    |  41.154 μs (5%) |         |   3.31 KiB (1%) |         108 |
| `["ALTRO", "escape"]`        |   8.907 ms (5%) |         |   1.65 MiB (1%) |         840 |
| `["ALTRO", "parallel_park"]` |   1.182 ms (5%) |         |   1.27 MiB (1%) |         931 |
| `["ALTRO", "pendulum"]`      |   1.030 ms (5%) |         | 333.46 KiB (1%) |         643 |
| `["iLQR", "cartpole"]`       |   5.272 ms (5%) |         |                 |             |
| `["iLQR", "double_int"]`     |  21.497 μs (5%) |         |                 |             |
| `["iLQR", "parallel_park"]`  | 316.800 μs (5%) |         |                 |             |
| `["iLQR", "pendulum"]`       |   2.160 ms (5%) |         |                 |             |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["ALTRO"]`
- `["iLQR"]`

## Julia versioninfo
```
Julia Version 1.3.1
Commit 2d5741174c (2019-12-30 21:36 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
      Ubuntu 18.04 LTS (tunnels-mlk X55)
  uname: Linux 4.15.0-1073-oem #83-Ubuntu SMP Mon Feb 17 11:21:18 UTC 2020 x86_64 x86_64
  CPU: Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz: 
                 speed         user         nice          sys         idle          irq
       #1-16  4800 MHz    6474851 s       4620 s    2418885 s   55166880 s          0 s
       
  Memory: 31.199615478515625 GB (16184.97265625 MB free)
  Uptime: 469074.0 sec
  Load Avg:  1.49560546875  1.421875  1.31787109375
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, skylake)
```