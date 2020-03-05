# Benchmark Report for *TrajectoryOptimization*

## Job Properties
* Time of benchmark: 5 Mar 2020 - 14:50
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

| ID                           | time            | GC time   | memory          | allocations |
|------------------------------|----------------:|----------:|----------------:|------------:|
| `["ALTRO", "3obs"]`          |   2.008 ms (5%) |           |   1.75 MiB (1%) |         620 |
| `["ALTRO", "acrobot"]`       |   5.448 ms (5%) |           |   2.71 MiB (1%) |         760 |
| `["ALTRO", "airplane"]`      |  35.520 ms (5%) |           |  14.49 MiB (1%) |         764 |
| `["ALTRO", "cartpole"]`      |   4.071 ms (5%) |           |   1.47 MiB (1%) |         382 |
| `["ALTRO", "double_int"]`    | 178.735 μs (5%) |           | 141.82 KiB (1%) |         298 |
| `["ALTRO", "escape"]`        |   8.707 ms (5%) |           |   1.81 MiB (1%) |        3135 |
| `["ALTRO", "parallel_park"]` |   1.600 ms (5%) |           |   1.39 MiB (1%) |         500 |
| `["ALTRO", "pendulum"]`      | 628.282 μs (5%) |           | 167.96 KiB (1%) |         134 |
| `["ALTRO", "quadrotor"]`     |  14.654 ms (5%) |           |   6.55 MiB (1%) |         141 |
| `["Ipopt", "3obs"]`          | 134.561 ms (5%) |           |  13.81 MiB (1%) |      195631 |
| `["Ipopt", "acrobot"]`       | 213.069 ms (5%) |           |  28.07 MiB (1%) |      390471 |
| `["Ipopt", "airplane"]`      |  23.027 ms (5%) |           |  15.24 MiB (1%) |      303371 |
| `["Ipopt", "cartpole"]`      | 128.645 ms (5%) |           |  18.44 MiB (1%) |      258044 |
| `["Ipopt", "double_int"]`    |  12.263 ms (5%) |           |   1.26 MiB (1%) |       19791 |
| `["Ipopt", "escape"]`        |    6.129 s (5%) |           |  48.68 MiB (1%) |      792009 |
| `["Ipopt", "parallel_park"]` |  82.026 ms (5%) |           |  11.86 MiB (1%) |      166128 |
| `["Ipopt", "pendulum"]`      |  73.166 ms (5%) |           |   8.34 MiB (1%) |      100972 |
| `["Ipopt", "quadrotor"]`     |   28.041 s (5%) | 81.112 ms | 937.59 MiB (1%) |    12061654 |
| `["iLQR", "acrobot"]`        |   5.538 ms (5%) |           |                 |             |
| `["iLQR", "airplane"]`       |  58.167 ms (5%) |           |                 |             |
| `["iLQR", "cartpole"]`       |   6.215 ms (5%) |           |                 |             |
| `["iLQR", "double_int"]`     |  23.719 μs (5%) |           |                 |             |
| `["iLQR", "parallel_park"]`  |   1.433 ms (5%) |           |                 |             |
| `["iLQR", "pendulum"]`       | 650.677 μs (5%) |           |                 |             |
| `["iLQR", "quadrotor"]`      |  18.138 ms (5%) |           |                 |             |

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
       #1-16  4801 MHz    1867080 s       4053 s     388257 s   46544699 s          0 s
       
  Memory: 31.199615478515625 GB (21775.10546875 MB free)
  Uptime: 30639.0 sec
  Load Avg:  1.43359375  1.20458984375  1.06982421875
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, skylake)
```