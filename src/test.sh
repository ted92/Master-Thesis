#!/bin/bash
	for i in {1..100}; do
		
		python observe_new.py -d
		echo $i
	done
