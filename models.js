const models = {
  'candy-8': {
    'input1': ['float32', [1, 3, 224, 224], 0.5],
  },

  'densenet-9': {
    'data_0': ['float32', [1, 3, 224, 224], 0.5],
  },

  'efficientnet-lite4-11': {
    'images:0': ['float32', [1, 224, 224, 3], 0.5],
  },

  'emotion-ferplus-8': {
    'Input3': ['float32', [1, 1, 64, 64], 0.5],
  },

  'inception-v1-12': {
    'data_0': ['float32', [1, 3, 224, 224], 0.5],
  },

  'mobilenetv2-7': {
    'input': ['float32', [1, 3, 224, 224], 0.5],
  },

  'mobilenetv2-10': {
    'input': ['float32', [1, 3, 224, 224], 0.5],
  },

  'resnet50-v1-12': {
    'data': ['float32', [1, 3, 224, 224], 0.5],
  },

  'segment-anything-vit-h-static-shapes-origin-im-size-initializer-optimized-float32': {
    'image_embeddings': ['float32', [1, 3, 224, 224], 0.5],
    'point_coords': ['float32', [1, 2, 2], [327.1111, 426.875, 241.77777, 341.5]],
    'point_labels': ['float32', [1, 2], [0, 1]],
    'mask_input': ['float32', [1, 1, 256, 256], 0],
    'has_mask_input': ['float32', [1], 1],
  },

  'squeezenet1.1-7': {
    'data': ['float32', [1, 3, 224, 224], 0.5],
  },

  // 't5-encoder-12'

  'tinyyolov2-8': {
    'image': ['float32', [1, 3, 416, 416], 0.5],
  },

  'yolo': {
    'image': ['float32', [1, 3, 416, 416], 0.5],
  },

  // no corresponding models
  'sam-h': {
    'image_embeddings': ['float32', [1, 256, 64, 64], 0.5],
    'point_coords': ['float32', [1, 2, 2], [327.1111, 426.875, 241.77777, 341.5]],
    'point_labels': ['float32', [1, 2], [0, 1]],
    'mask_input': ['float32', [1, 1, 256, 256], 0],
    'has_mask_input': ['float32', [1], 1],
    // orig_im_size is optimized out for this model
    //'orig_im_size': ['float32', [2], [1200., 1800.]],
  },

  // TODO: Convert to actual float16 values. We're just estimating perf with this one, not correctness.
  'sam-h-16': {
    'image_embeddings': ['float16', [1, 256, 64, 64], 0.5],
    'point_coords': ['float16', [1, 2, 2], [327, 426, 241, 341]],
    'point_labels': ['float16', [1, 2], [0, 2]],
    'mask_input': ['float16', [1, 1, 256, 256], 0],
    'has_mask_input': ['float16', [1], 1],
    // orig_im_size is optimized out for this model
    //'orig_im_size': ['float16', [2], [1200., 1800.]],
  },

  'onnx-add': {
    'A': ['float32', [5], 1],
    'B': ['float32', [5], 1],
  }
}