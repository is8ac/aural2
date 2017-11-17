#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8 9
do
  arecord -D plughw:$i --r 16000 -t raw -f S16_LE -c 1 - | aural2
done
