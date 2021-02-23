#!/bin/bash
cd build
for detector in {0..6}
do
	for descriptor in {0..5}
	do
	./3D_object_tracking $detector $descriptor 0
	done
done
