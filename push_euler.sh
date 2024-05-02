#!/usr/bin/env bash

if [ $(hostname) = "xps" ]; then
    LOCAL_PROJ_DIR=~/approximate_optimal_control/
    echo TODO: make sure this is correct as well!
    exit
else
    # ~ is /nas/dbalduin :(
    LOCAL_PROJ_DIR=/local/home/dbalduin/approximate_optimal_control/
fi

# echo syncing from $LOCAL_PROJ_DIR to euler...
# 
# rsync -av --dry-run --exclude-from=.gitignore --exclude='.git*'  $LOCAL_PROJ_DIR/ dbalduin@euler.ethz.ch:/cluster/home/dbalduin/approximate_optimal_control
# 
# echo
# echo "^^^ this was the dry run ^^^"
# echo -n "push these files for real? [y/n] "
# 
# read answer
# 
# if [ $answer = y ]; then
#     rsync -av --exclude-from=.gitignore --exclude='.git*'  $LOCAL_PROJ_DIR/ dbalduin@euler.ethz.ch:/cluster/home/dbalduin/approximate_optimal_control
# else
#     echo not doing anything. goodbye
# fi


# kind of dumb to do a dry run and ask every time. trust the user :) 
# exclude also the git directory (not in gitignore...) and the euler pull/push
# scripts which don't make sense running on euler

rsync -av --exclude-from=.gitignore --exclude='.git*' --exclude='*euler.sh'  $LOCAL_PROJ_DIR/ dbalduin@euler.ethz.ch:/cluster/home/dbalduin/approximate_optimal_control
