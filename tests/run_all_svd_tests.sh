#!/usr/bin/env bash
# Quick test script to validate all SVD methods after rebuild

set -e

echo "=================================="
echo "ACTIONet SVD Methods Test Suite"
echo "=================================="
echo ""

cd "$(dirname "$0")"

echo "Running SVD methods comparison test..."
echo "--------------------------------------"
python test_svd_methods.py
echo ""

echo "Running sparse vs dense validation test..."
echo "------------------------------------------"
python test_svd_sparse_vs_dense.py
echo ""

echo "=================================="
echo "All tests completed!"
echo "=================================="
echo ""
echo "Check the following output files:"
echo "  - svd_methods_comparison.png"
echo "  - svd_singular_values.png"
echo "  - svd_sparse_vs_dense_comparison.png"
echo "  - svd_sparse_vs_dense_summary.png"
echo ""
