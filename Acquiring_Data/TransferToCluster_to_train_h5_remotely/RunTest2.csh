#!/bin/tcsh                                                                                                                                           

echo "First arg: $1"
echo "First arg: $2" 
mkdir -p $1
cp *.py $1
touch $1/FirstRun.log
touch $1/GenerateDeform.log
touch $1/RunWDef.log
cp $4_* $1
mkdir $1/data
mkdir $1/data/data_temp
cd $1
python3 run_NNmasks_f.py $4_CZANet_Final.h5 FirstRun.log 0 150 0 0 $2 $3
rm -r data/data_temp/zmdir
cp $4_CZANet_Final.h5 $4_CZANet_Final-80.h5
python3 run_NNmasks_f.py $4_CZANet_Final-80.h5 GenerateDeform.log 1 1 1 0 3 81 0
rm -r data/data_temp/zmdir
cp $4_CZANet_Final-80.h5 $4_CZANet_Final-80WDef.h5
python3 run_NNmasks_f.py $4_CZANet_Final-80WDef.h5 RunWDef.log 0 80 1 1 0 
rm -r data/data_temp/zmdir

