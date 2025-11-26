#!/bin/bash
set -e

echo "Running ruff check..."
ruff check . --fix

echo "Running ruff format..."
ruff format .
