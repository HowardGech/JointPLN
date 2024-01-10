#!/bin/bash

# Get the directory of the script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change the current working directory
cd "$script_dir"

git add .

echo 'Enter the commit message:'
read commitMessage

git commit -m "$commitMessage"

git push

if [ $? -eq 0 ]; then
    echo "Push succeeds."
else
    echo "Push failed with exit status $?."
fi

echo 'Press Enter to Exit'

read

kill `ps -A | grep -w Terminal.app | grep -v grep | awk '{print $1}'`
