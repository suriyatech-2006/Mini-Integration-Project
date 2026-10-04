#ifndef MODEL_H
#define MODEL_H

// Template header.
// Run Smart_Light_TinyML.ipynb, create light_level_classifier.tflite,
// then run convert_model.py to replace this array with real model bytes.

alignas(8) const unsigned char g_model[] = {
  0x00
};

const unsigned int g_model_len = sizeof(g_model);

#endif
