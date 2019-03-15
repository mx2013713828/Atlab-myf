#!/bin/bash
dir_a=$1
dir_b=$2
output=$3
label=$4
python ronghe.py $dir_a $dir_b $output

sh split_date.sh $output $label

qshell account 

for file in `ls ./$label/`
do
    qshell fput myf666 $label/$file $label/$file;
done


