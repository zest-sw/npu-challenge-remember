# /home/ycc/src/remembr/env_set.sh
export TT_METAL_HOME=/tt-metal
export BOS_METAL_HOME=$TT_METAL_HOME/tt_metal/third_party/bos-metal
export PYTHONPATH=$TT_METAL_HOME:$BOS_METAL_HOME:$PYTHONPATH
source $TT_METAL_HOME/python_env/bin/activate
