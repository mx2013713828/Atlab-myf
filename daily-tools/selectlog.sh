for file in `cat log.lst`;
do
    ./qrsctl get qpulp-log $file $file
    cat $file|grep -E 'uid":1366779496\uid":1380772729\uid"1380969784' >yz-wpt-qtt-$file.json
    rm $file
done
