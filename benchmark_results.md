# Benchmark Report for *TrajectoryOptimization*

## Job Properties
* Time of benchmark: 14 Dec 2018 - 16:3
* Package commit: dirty
* Julia commit: 5d4eac
* Julia command flags: None
* Environment variables: None

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                                | time            | GC time    | memory          | allocations |
|---------------------------------------------------|----------------:|-----------:|----------------:|------------:|
| `["dubinscar", "parallel park", "constrained"]`   | 496.873 ms (5%) |  52.329 ms | 254.19 MiB (1%) |     2947876 |
| `["dubinscar", "parallel park", "unconstrained"]` | 283.216 ms (5%) |  35.253 ms | 155.74 MiB (1%) |     1445517 |
| `["pendulum", "constrained"]`                     | 775.819 ms (5%) | 133.073 ms | 533.21 MiB (1%) |     7552176 |
| `["pendulum", "unconstrained"]`                   | 120.869 ms (5%) |  15.760 ms |  76.56 MiB (1%) |      929202 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dubinscar", "parallel park"]`
- `["pendulum"]`

## Julia versioninfo
```
Julia Version 1.0.0
Commit 5d4eaca0c9 (2018-08-08 20:58 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
      Ubuntu 16.04.5 LTS
  uname: Linux 4.15.0-42-generic #45~16.04.1-Ubuntu SMP Mon Nov 19 13:02:27 UTC 2018 x86_64 x86_64
  CPU: Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz: 
              speed         user         nice          sys         idle          irq
       #1  2860 MHz     767212 s        599 s     179493 s    2997223 s          0 s
       #2  2840 MHz     758401 s        801 s     179961 s    1288138 s          0 s
       #3  2842 MHz     726560 s        655 s     184849 s    1306409 s          0 s
       #4  2837 MHz     770200 s        478 s     181288 s    1267288 s          0 s
       
  Memory: 7.538402557373047 GB (614.234375 MB free)
  Uptime: 81616.0 sec
  Load Avg:  3.185546875  2.25048828125  1.9375
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.0 (ORCJIT, skylake)
```