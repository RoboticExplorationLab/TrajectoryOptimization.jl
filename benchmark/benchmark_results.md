# Benchmark Report for *TrajectoryOptimization*

## Job Properties
* Time of benchmark: 2 Mar 2020 - 19:52
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
| `["ALTRO", "3obs"]`          | 961.085 μs (5%) |         | 980.07 KiB (1%) |         883 |
| `["ALTRO", "cartpole"]`      |   4.763 ms (5%) |         |  22.13 KiB (1%) |         508 |
| `["ALTRO", "double_int"]`    |  41.575 μs (5%) |         |   3.31 KiB (1%) |         108 |
| `["ALTRO", "escape"]`        |   9.003 ms (5%) |         |   1.65 MiB (1%) |         840 |
| `["ALTRO", "parallel_park"]` |   1.180 ms (5%) |         |   1.27 MiB (1%) |         931 |
| `["ALTRO", "pendulum"]`      |   1.035 ms (5%) |         | 333.46 KiB (1%) |         643 |
| `["Ipopt", "3obs"]`          | 179.453 ms (5%) |         |   1.99 MiB (1%) |       36955 |
| `["Ipopt", "cartpole"]`      | 119.807 ms (5%) |         |   2.01 MiB (1%) |       36644 |
| `["Ipopt", "double_int"]`    |  15.654 ms (5%) |         | 712.05 KiB (1%) |       13875 |
| `["Ipopt", "escape"]`        |    6.068 s (5%) |         |  18.99 MiB (1%) |      396811 |
| `["Ipopt", "parallel_park"]` |  59.814 ms (5%) |         |   1.60 MiB (1%) |       27414 |
| `["Ipopt", "pendulum"]`      | 255.768 ms (5%) |         |   1.87 MiB (1%) |       37452 |
| `["iLQR", "cartpole"]`       |   5.233 ms (5%) |         |                 |             |
| `["iLQR", "double_int"]`     |  21.352 μs (5%) |         |                 |             |
| `["iLQR", "parallel_park"]`  | 303.221 μs (5%) |         |                 |             |
| `["iLQR", "pendulum"]`       |   2.159 ms (5%) |         |                 |             |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["ALTRO"]`
- `["Ipopt"]`
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
       #1-16  4228 MHz    7210025 s       4644 s    2612538 s   67508375 s          0 s
       
  Memory: 31.199615478515625 GB (16205.56640625 MB free)
  Uptime: 477402.0 sec
  Load Avg:  1.5234375  1.49365234375  1.2900390625
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, skylake)
```