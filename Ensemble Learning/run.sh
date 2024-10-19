#!/bin/sh

echo "For results of Adaboost"
python3 AdaBoost.py

echo "For results of  bagged tree"
python3 BaggedTree.py

echo "For results of biasvariance"
python3 VarBias.py

echo "For results of randomforest"
python3 RandomForest.py

echo "Bias Variance Random"
python3 estimateBiasVar.py
