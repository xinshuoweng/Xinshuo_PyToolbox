#!/usr/bin/env sh

modelfolder=$1

for ext in "jpg" "png" "jpeg" "bmp"
do
	find $modelfolder/ -type f -name "*.$ext" -delete
done