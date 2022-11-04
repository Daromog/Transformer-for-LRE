# Transformer-for-LRE
This code was used to perform Language recognition using phonotactic information obtained from speech signals

The transformer encoder implements sliding attention windows from "LongFormer" and "BigBird" to manage long input sequences. We take advantage of the attention mechanism
to find discriminative combination of characters to differentiate between similar languages. This model was fused with a pure acoustic system based on MFCC-SDC-iVectors.
