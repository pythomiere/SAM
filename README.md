## About

To run the file, run `predict.py`, other files are used for training.
For dataset, I used SST-2 (I tried UCI but SST-2 performs better).  
This model uses bag of words and has ~80% accuracy, which is similar to other bag of words model.

Given a sentence, the model will output a number between 0 and 1, the higher the number, the more likely the model thinks this sentence is positive. Vice versa.

| Hyperparameter | Value |
|-----------|-------|
| Vocabulary size | 8001 (8000 words + `<UNK>` at index 0) |
| Embedding dim | 64 |
| Hidden layer size | 32 |
| Output size | 1 |

## Files
 
`vocab.bin` stores the vocabulary, each word entry is exactly 64 bytes and null-padded. 
So to find word at index `i` , `offset = 4 + (i * 64)`.

All values at `weight.bin` are 32-bits floats. 

Here are the format: 

```
Bytes 0-15:     Header (4 x uint32_t)
                  [0] vocab_size  = 8001
                  [1] embed_dim   = 64
                  [2] hidden_dim  = 32
                  [3] output_dim  = 1

Bytes 16-2,048,271:   Embedding matrix
                       Shape: vocab_size × embed_dim = 8001 × 64
                       Total: 512,064 floats
                       Layout: row-major. Row i is the embedding for word index i.

Bytes 2,048,272-2,056,463:  FC1 weights
                             Shape: embed_dim × hidden_dim = 64 × 32
                             Total: 2,048 floats
                             Layout: row-major. Already transposed for C.

Bytes 2,056,464-2,056,591:  FC1 bias
                             Shape: hidden_dim = 32
                             Total: 32 floats

Bytes 2,056,592-2,056,719:  FC2 weights
                             Shape: hidden_dim × output_dim = 32 × 1
                             Total: 32 floats
                             Layout: row-major. Already transposed for C.

Bytes 2,056,720-2,056,723:  FC2 bias
                             Shape: output_dim = 1
                             Total: 1 float
```
## Inference

1. Tokenize: lowercase -> strip `string.punctuation` -> split on whitespace.
   Unknown words map to index 0, the same goes for empty.
2. Embed: look up each token index in the embedding matrix, average the rows -> 64 floats.
3. FC1: ReLU
4. FC2: sigmoid
5. Output: float in (0, 1), > 0.5 -> Positive.
