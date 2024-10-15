#!/bin/sh

echo "Avg prediction errors for car"
python3 car_hw1.py

echo "Avg prediction errors for bank -unknown as particular attribute value"
python3 bank_replace.py

echo "Avg prediction errors for bank -unknown as attribute value missing"
python3 bank_no_replace.py