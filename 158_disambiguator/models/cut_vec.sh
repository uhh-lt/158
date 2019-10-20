#!/usr/bin/env bash

for lang in en ru
do
    echo "LANG $lang"
    echo "Copying cc.$lang.300.vec.gz file..."
    cp $lang/cc.$lang.300.vec.gz $lang/cc.$lang.300.vec.2.gz
    echo "UnGZ cc.$lang.300.vec.2.gz file..."
    gzip -d $lang/cc.$lang.300.vec.2.gz
    echo "Cutting cc.$lang.300.vec.2 to cc.$lang.100k.300.vec file..."
    head -100001 $lang/cc.$lang.300.vec.2 > $lang/cc.$lang.100k.300.vec
    echo "Deleting cc.$lang.300.vec.2 file..."
    rm $lang/cc.$lang.300.vec.2
    echo "Replacing 1st line in cc.$lang.100k.300.vec file..."
    sed -i "1s/.*/100000 300/" $lang/cc.$lang.100k.300.vec
    echo "Archiving cc.$lang.100k.300.vec file..."
    gzip $lang/cc.$lang.100k.300.vec
done
