#!/bin/bash
# build.sh SOURCE OUT -- compiles a lab program against the ENGRAM sources (same flags class as the tree)
SRC="$(cd "$(dirname "$0")/../../src" && pwd)"
exec gcc -O2 -std=c99 -ffp-contract=off -I$SRC -o "$2" "$1" $SRC/engram_core.c $SRC/engram_alloc.c \
  $SRC/engram_plat.c $SRC/engram_log.c $SRC/engram_buf.c $SRC/engram_rng.c $SRC/engram_text.c \
  $SRC/engram_enc.c -lm -lpthread
