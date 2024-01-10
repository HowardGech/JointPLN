#!/bin/bash

# Get the directory of the script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change the current working directory
cd "$script_dir"

git pull

if [ $? -eq 0 ]; then
    echo "Pull succeeds."
else
    echo "Pull failed with exit status $?."
fi

echo 'Press Enter to Exit'

read

kill `ps -A | grep -w Terminal.app | grep -v grep | awk '{print $1}'`
