#!/bin/bash
inputfile=$1
label=$2
for date in `cat $inputfile'.json'|grep -e '2018-[0-9]\{2\}-[0-9]\{2\}' -o|sort -u` 

do  
    cat $inputfile'.json'|grep 'nrop-'$date > $label'/nrop-'$date'-'$label'-intersect'.json;
done

for date in `cat $inputfile'_diff.json'|grep -e '2018-[0-9]\{2\}-[0-9]\{2\}' -o|sort -u` 
do
    cat $inputfile'_diff.json'|grep 'nrop-'$date > $label'/nrop-'$date'-'$label'-diff'.json;
done


