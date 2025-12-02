#!/bin/env bash
ERR ()
{
    echo error occured in $@ >&2
    exit 1
}

../../bin/compress c fl $1 $1.fle || ERR fle compression
../../bin/compress d fl $1.fle $1_fle_decomp || ERR fle decompression
cmp $1 $1_fle_decomp || ERR results dont match
echo fle test succeded
