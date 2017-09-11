#!/bin/bash
	for i in {1..1000}; do
		
		python observe_new.py -d
		echo $i
	done
