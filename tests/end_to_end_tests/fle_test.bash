#!/bin/env bash
ERR ()
{
    echo error occured in $@ >&2
    exit 1
}

../../bin/frle -fco $1.fle $1 || ERR fle compression
../../bin/frle -fdo $1_fle_decomp $1.fle || ERR fle decompression
cmp $1 $1_fle_decomp || ERR results dont match
echo fle test succeded
