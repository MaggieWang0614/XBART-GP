#! /bin/bash
echo Building python
cd ../python/
bash build_py.sh -d
cd ../tests/
python test_gp.py

