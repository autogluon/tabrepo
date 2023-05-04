This C++ implementation of ROC_AUC is based on the code from https://github.com/diditforlulz273/fastauc,
with some additional optimizations.

You can find the original MIT license included within this folder (`LICENSE.md`).

## Compile

To compile, run `./compile.sh`.

## Changes

To accelerate the code beyond the original implementation, `sample_weights` support was removed.
Additionally, the return type was changed from `float` to `double` for enhanced precision.
