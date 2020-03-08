# Benchmark Report for *TrajectoryOptimization*

## Job Properties
* Time of benchmark: 7 Mar 2020 - 15:53
* Package commit: 4a7fc9
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

| ID                           | time            | GC time    | memory          | allocations |
|------------------------------|----------------:|-----------:|----------------:|------------:|
| `["ALTRO", "3obs"]`          |   1.989 ms (5%) |            |   1.75 MiB (1%) |         620 |
| `["ALTRO", "acrobot"]`       |   5.178 ms (5%) |            |   2.71 MiB (1%) |         760 |
| `["ALTRO", "airplane"]`      |  33.546 ms (5%) |            |  14.49 MiB (1%) |         764 |
| `["ALTRO", "cartpole"]`      |   4.105 ms (5%) |            |   1.47 MiB (1%) |         382 |
| `["ALTRO", "double_int"]`    | 171.476 μs (5%) |            | 141.82 KiB (1%) |         298 |
| `["ALTRO", "escape"]`        |   11.242 s (5%) | 409.991 ms |   2.28 GiB (1%) |    58010678 |
| `["ALTRO", "parallel_park"]` |   1.566 ms (5%) |            |   1.39 MiB (1%) |         500 |
| `["ALTRO", "pendulum"]`      | 622.282 μs (5%) |            | 167.96 KiB (1%) |         134 |
| `["ALTRO", "quadrotor"]`     |    5.835 s (5%) | 274.637 ms |   1.41 GiB (1%) |    37179149 |
| `["Ipopt", "3obs"]`          | 125.871 ms (5%) |            |  13.81 MiB (1%) |      195631 |
| `["Ipopt", "acrobot"]`       | 196.511 ms (5%) |            |  28.07 MiB (1%) |      390471 |
| `["Ipopt", "airplane"]`      |    9.230 s (5%) | 560.205 ms |   2.46 GiB (1%) |    73481145 |
| `["Ipopt", "cartpole"]`      | 119.043 ms (5%) |            |  18.44 MiB (1%) |      258044 |
| `["Ipopt", "double_int"]`    |  11.418 ms (5%) |            |   1.26 MiB (1%) |       19791 |
| `["Ipopt", "escape"]`        |   17.721 s (5%) | 348.981 ms |   1.91 GiB (1%) |    51927469 |
| `["Ipopt", "parallel_park"]` |  76.228 ms (5%) |            |  11.85 MiB (1%) |      166128 |
| `["Ipopt", "pendulum"]`      |  66.899 ms (5%) |            |   8.34 MiB (1%) |      100972 |
| `["Ipopt", "quadrotor"]`     |   43.731 s (5%) |    1.342 s |   5.10 GiB (1%) |   134749401 |
| `["iLQR", "acrobot"]`        |   5.272 ms (5%) |            |                 |             |
| `["iLQR", "airplane"]`       |  54.547 ms (5%) |            |                 |             |
| `["iLQR", "cartpole"]`       |   5.995 ms (5%) |            |                 |             |
| `["iLQR", "double_int"]`     |  22.574 μs (5%) |            |                 |             |
| `["iLQR", "parallel_park"]`  |   1.384 ms (5%) |            |                 |             |
| `["iLQR", "pendulum"]`       | 621.362 μs (5%) |            |                 |             |
| `["iLQR", "quadrotor"]`      |  17.058 ms (5%) |            |                 |             |

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
       #1-16  4954 MHz    4673384 s       5560 s    1002793 s   15591194 s          0 s
       
  Memory: 31.199615478515625 GB (19895.92578125 MB free)
  Uptime: 207211.0 sec
  Load Avg:  1.0615234375  1.08154296875  0.9375
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, skylake)
```