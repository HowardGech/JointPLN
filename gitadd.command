#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
git add .

echo 'Enter the commit message:'
read commitMessage

git commit -m "$commitMessage"

branch='main'
echo 'Enter the name of the branch. Default is main:'
read branch

git push origin $branch
echo 'Press Enter to exit'

read