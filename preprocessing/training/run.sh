#!/bin/bash

set -e

# надёжно вычисляем папку со скриптом
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🔁 Step 1: step_1.py"
python step_1.py
echo "✅ Done: step_1.py"

echo "🔁 Step 2: step_2.py"
python step_2.py
echo "✅ Done: step_1.py"

echo "🔁 Step 3: step_3.py"
python step_3.py
echo "✅ Done: step_3.py"

echo "🔁 Step 4: step_4.py"
python step_4.py
echo "✅ Done: step_4.py"

echo "🔁 Step 5: step_5.py"
python step_5.py
echo "✅ Done: step_5.py"

echo "🔁 Step 6: step_6.py"
python step_6.py
echo "✅ Done: step_6.py"

echo "🔁 Step 7: step_7.py"
python step_7.py
echo "✅ Done: step_7.py"

echo "🔁 Step 8: step_8.py"
python step_8.py
echo "✅ Done: step_8.py"

echo "🎉 ALL STEPS COMPLETED SUCCESSFULLY"