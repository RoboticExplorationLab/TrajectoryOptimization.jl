using PkgBenchmark

res = benchmarkpkg("TrajectoryOptimization")
export_markdown("benchmark/benchmark_results.md", res)
