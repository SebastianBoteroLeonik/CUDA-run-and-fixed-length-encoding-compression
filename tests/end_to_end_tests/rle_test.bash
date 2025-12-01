#!/bin/env bash
ERR ()
{
    echo error occured in $@ >&2
    exit 1
}

../../bin/frle -rco $1.rle $1 || ERR rle compression
../../bin/frle -rdo $1_rle_decomp $1.rle || ERR rle decompression
cmp $1 $1_rle_decomp || ERR results dont match
echo rle test succeded
