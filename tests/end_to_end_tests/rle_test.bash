#!/bin/env bash
ERR ()
{
    echo error occured in $@ >&2
    exit 1
}

../../bin/compress c rl $1 $1.rle || ERR rle compression
../../bin/compress d rl $1.rle $1_rle_decomp || ERR rle decompression
cmp $1 $1_rle_decomp || ERR results dont match
echo rle test succeded
