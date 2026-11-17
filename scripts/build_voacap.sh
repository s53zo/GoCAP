#!/usr/bin/env bash
set -eu
# Requires git, make, autoconf, automake and gfortran (macOS: brew install gcc autoconf automake).
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_source="$task_root/data/voacap-study/voacapl"
task_prefix="$task_root/data/voacap-study/install"
mkdir -p "$(dirname "$task_source")"
if [ ! -d "$task_source/.git" ]; then
  git clone https://github.com/jawatson/voacapl "$task_source"
fi
cd "$task_source"
git checkout c12a98b348c39f36f2659b8b3aa97b86284c8a1d
autoreconf -fi
./configure --prefix="$task_prefix" FFLAGS='-O2 -std=legacy -fallow-argument-mismatch' FCFLAGS='-O2 -std=legacy -fallow-argument-mismatch'
make -j4
make install
# Do not run makeitshfbc: benchmark copies coefficients into an isolated temporary run directory.
