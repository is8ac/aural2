#!/bin/bash
arecord -D plughw:3 -r 16000 -t raw -f S16_LE -c 1 - | aural2
