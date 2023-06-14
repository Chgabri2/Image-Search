#!/bin/bash
for i in {5..10}
do  
    A=2000

    /Users/gab/Documents/GitHub/image-search/.venv/bin/python /Users/gab/Documents/GitHub/image-search/funcs.py  `expr $A \* $i` 
done
