#!/bin/bash

GPU_DEVICE=0
EPOCHS=2
rm -f test.txt
for BUDGET in 7168.00 10035.20 13619.20
do
    python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 16 -e $EPOCHS -budget $BUDGET
    mv  test.txt pythia_2_$BUDGET
done

for BUDGET in 8192.00 11059.20 14643.20
do
    python ./train_llm.py -net facebook/opt-350m -gpu -b 8 -e $EPOCHS -budget $BUDGET
    mv  test.txt opt_2_$BUDGET
done

for BUDGET in 6144.00 7823.36 9922.56
do
    python ./train_llm.py -net openai-community/gpt2 -gpu -b 8 -e $EPOCHS -budget $BUDGET
     mv  test.txt gpt_2_$BUDGET
done





