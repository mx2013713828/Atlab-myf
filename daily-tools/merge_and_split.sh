#!/bin/bash
dir_a=$1
dir_b=$2
output=$3
label=$4
python ronghe.py $dir_a $dir_b $output

sh split_date.sh $output $label

qshell account XBhrm_cIe71aI4HZPj7oBZm-uYPa4D3pzVfNw5LB -2uQxaJFqFcCKUjyCvEkvphzK5JOxakj5aWXyi6u

for file in `ls $label/`
do
    qshell fput qpulp-annotations $file $label/$file;
done
