#!/bin/bash
set -e  # stop on error

echo "=== Running data tests ==="
pytest test_data.py

echo "=== Evaluating model ==="
python evaluate.py

echo "=== Running API tests ==="
pytest test_api.py
