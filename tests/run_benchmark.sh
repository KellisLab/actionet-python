#!/bin/bash
# Convenience script to run SVD algorithm benchmarks

# Default values
COMPONENTS=30
RUNS=3
DETAILED_MEMORY=""

# Help message
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run comprehensive SVD algorithm benchmarks for ACTIONet.

OPTIONS:
    -c, --components NUM    Number of components to compute (default: 30)
    -r, --runs NUM          Number of benchmark runs per test (default: 3)
    -m, --detailed-memory   Enable detailed memory profiling (requires memory_profiler)
    -h, --help              Show this help message

EXAMPLES:
    $(basename "$0")                          # Run with defaults
    $(basename "$0") -c 50 -r 5               # 50 components, 5 runs
    $(basename "$0") --detailed-memory        # Enable detailed memory profiling

OUTPUT:
    - benchmark_svd_results.csv    Detailed results in CSV format
    - benchmark_svd_results.png    Visualization with 4 subplots
    - Console output               Summary table and statistics

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--components)
            COMPONENTS="$2"
            shift 2
            ;;
        -r|--runs)
            RUNS="$2"
            shift 2
            ;;
        -m|--detailed-memory)
            DETAILED_MEMORY="--detailed-memory"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if psutil is installed
if ! python -c "import psutil" 2>/dev/null; then
    echo "Warning: psutil not installed. Memory tracking will be limited."
    echo "Install with: pip install psutil"
    echo ""
fi

# Run benchmark
echo "Running SVD Algorithm Benchmark..."
echo "Components: $COMPONENTS"
echo "Runs: $RUNS"
echo ""

python benchmark_svd_algorithms.py \
    --components "$COMPONENTS" \
    --runs "$RUNS" \
    $DETAILED_MEMORY

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Benchmark completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - benchmark_svd_results.csv (detailed results)"
    echo "  - benchmark_svd_results.png (visualization)"

    # Open visualization if on macOS
    if [[ "$OSTYPE" == "darwin"* ]] && [ -f "benchmark_svd_results.png" ]; then
        echo ""
        read -p "Open visualization? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            open benchmark_svd_results.png
        fi
    fi
else
    echo ""
    echo "Benchmark failed. Check the error messages above."
    exit 1
fi
