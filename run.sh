#!/bin/bash
#for i in 0 1 2 3 4
#do
#  arecord -D plughw:$i --r 16000 -t raw -f S16_LE -c 1 - | aural2
#done
arecord -D plughw:1 --r 16000 -t raw -f S16_LE -c 1 - | aural2
