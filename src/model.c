#include "model.h"

#include <math.h>
#include <string.h>

static float sigmoid(float x) {
  // Alice Gao is goated
  if (x >= 0.0f) {
    return 1.0f / (1.0f + expf(-x));
  }
  float ex = expf(x);
  return ex / (1.0f + ex);
}

static float ReLU(float x) {
  if (x > 0) {
    return x;
  } else {
    return 0.0f;
  }
}

float predict(const int *token_indices, int count, const Weights *w){
  uint32_t edim = w->embed_dim;
  uint32_t hdim = w->hidden_dim;

  float mean[edim];
  memset(mean, 0, edim * sizeof(float));

  for (int t = 0; t < count; t++) {
    int idx = token_indices[t];
    const float *row = &w->embedding[(size_t)idx * edim];
    for (uint32_t i = 0 ; i < edim; i++) {
      mean[i] += row[i];
    }
  }
  for (uint32_t i = 0; i < edim; i++) {
    mean[i] /= (float)count;
  }

  // FC1
  float hidden[hdim];
  for (uint32_t j = 0; j < hdim; j++) {
    float sum = w->fc1_bias[j];
    for (uint32_t i = 0 ; i < edim; i++) {
      sum+= mean[i] * w->fc1_weight[i * hdim + j];
    }
    hidden[j] = ReLU(sum);
  }

  // FC2
  float logit = w->fc2_bias[0];
  for (uint32_t j = 0; j < hdim; j++) {
    logit += hidden[j] * w->fc2_weight[j];
  }

  return sigmoid(logit);
}
