# Benchmark Report for *TrajectoryOptimization*

## Job Properties
* Time of benchmark: 2 Mar 2020 - 17:12
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

| ID                           | time            | GC time    | memory          | allocations |
|------------------------------|----------------:|-----------:|----------------:|------------:|
| `["ALTRO", "3obs"]`          | 966.225 μs (5%) |            | 980.07 KiB (1%) |         883 |
| `["ALTRO", "cartpole"]`      |   4.795 ms (5%) |            |  22.13 KiB (1%) |         508 |
| `["ALTRO", "double_int"]`    |  41.746 μs (5%) |            |   3.31 KiB (1%) |         108 |
| `["ALTRO", "escape"]`        |   14.039 s (5%) | 511.091 ms |   2.45 GiB (1%) |    61866024 |
| `["ALTRO", "parallel_park"]` |   1.155 ms (5%) |            |   1.27 MiB (1%) |         931 |
| `["ALTRO", "pendulum"]`      |   1.038 ms (5%) |            | 333.46 KiB (1%) |         643 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["ALTRO"]`

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
       #1-16  4800 MHz    6346001 s       4606 s    2388991 s   53309124 s          0 s
       
  Memory: 31.199615478515625 GB (15796.4921875 MB free)
  Uptime: 467809.0 sec
  Load Avg:  1.44189453125  1.21630859375  1.1982421875
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, skylake)
```