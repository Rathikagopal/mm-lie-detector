#!/bin/bash

JUPYTER_PORT=7999

if [[ $(ss -ln src :$JUPYTER_PORT | grep -Ec -e "\<$JUPYTER_PORT\>") -eq 0 ]]; then
    exec jupyter notebook --port 7999 --ip='*' --NotebookApp.token='' --NotebookApp.password='' /workspace &> /dev/null &
fi
