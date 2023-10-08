const models = {
  'albert-base-v2': 'bert64', // tjs/albert-base-v2/onnx/model.onnx. TODO: NaN
  'bart-large-cnn-encoder': 'bert64', // tjs/facebook/bart-large-cnn/onnx/encoder_model.onnx
  'bert-base-cased': 'bert64', // tjs/bert-base-cased/onnx/model.onnx
  'bert-base-uncased': 'bert64', // tjs/bert-base-uncased/onnx/model.onnx
  'candy-8': 'img224', // webnn. If the value is set to 0.5, conformance test would fail.
  'clip-vit-base-patch16': 'clip', // tjs/openai/clip-vit-base-patch16/onnx/model.onnx
  'densenet-9': 'img224', // webnn
  'detr-resnet-50': 'img224', // tjs/facebook/detr-resnet-50/onnx/model.onnx. TODO: conformance fails
  'dino-vitb16': 'img224', // tjs/facebook/dino-vitb16/onnx/model.onnx
  'distilbert-base-uncased': 'bert64', // tjs/distilbert-base-uncased/onnx/model.onnx
  'distilgpt2': 'llm-decoder', // tjs/gpt2/onnx/decoder_model_merged.onnx. TODO: NaN
  'efficientnet-lite4-11': { 'images:0': ['float32', 'random', [1, 224, 224, 3]] }, // webnn
  'emotion-ferplus-8': { 'Input3': ['float32', 'random', [1, 1, 64, 64]] }, // webnn
  'gpt2': 'llm-decoder', // tjs/gpt2/onnx/decoder_model_merged.onnx. TODO: NaN
  'mobilenetv2-7': 'img224', // obsolete
  'mobilenetv2-10': 'img224', // obsolete
  'mobilenetv2-12': 'img224', // webnn
  'resnet50-v1-12': 'img224', // obsolete
  'resnet50-v2-7': 'img224', // webnn
  't5-small-decoder': 't5-decoder', // tjs/t5-small/onnx/decoder_model_merged.onnx
  't5-small-encoder': 't5-encoder', // tjs/t5-small/onnx/encoder_model.onnx
  'tinyyolov2-8': { 'image': ['float32', 'random', [1, 3, 416, 416]] }, // webnn
  'whisper-tiny-decoder': 'whisper-decoder', // tjs/openai/whisper-tiny/onnx/decoder_model_merged.onnx
  'whisper-tiny-encoder': { 'input_features': ['float32', 'random', [1, 80, 3000]] }, // tjs/openai/whisper-tiny/onnx/encoder_model.onnx


  // todo

  'sam-h': {
    'image_embeddings': ['float32', 0.5, [1, 3, 224, 224]],
    'point_coords': ['float32', [327.1111, 426.875, 241.77777, 341.5], [1, 2, 2]],
    'point_labels': ['float32', [0, 1], [1, 2]],
    'mask_input': ['float32', 0, [1, 1, 256, 256]],
    'has_mask_input': ['float32', 1, [1]],
  },
  'sam-h-decoder': 'sam-decoder', // sam/sam-h. TODO: conformance fails

  'inception-v1-12': {
    'data_0': ['float32', 0.5, [1, 3, 224, 224]],
  },
  'squeezenet1.1-7': {
    'data': ['float32', 0.5, [1, 3, 224, 224]],
  },

  // 't5-encoder-12'
  'yolo': {
    'image': ['float32', 0.5, [1, 3, 416, 416]],
  },

  // no corresponding models
  'sam-h-xxx': {
    'image_embeddings': ['float32', 0.5, [1, 256, 64, 64]],
    'point_coords': ['float32', [327.1111, 426.875, 241.77777, 341.5], [1, 2, 2]],
    'point_labels': ['float32', [0, 1], [1, 2]],
    'mask_input': ['float32', 0, [1, 1, 256, 256]],
    'has_mask_input': ['float32', 1, [1]],
    // orig_im_size is optimized out for this model
    //'orig_im_size': ['float32', [2], [1200., 1800.]],
  },

  // TODO: Convert to actual float16 values. We're just estimating perf with this one, not correctness.
  'sam-h-16': {
    'image_embeddings': ['float16', 0.5, [1, 256, 64, 64]],
    'point_coords': ['float16', [327, 426, 241, 341], [1, 2, 2]],
    'point_labels': ['float16', [0, 2], [1, 2]],
    'mask_input': ['float16', 0, [1, 1, 256, 256]],
    'has_mask_input': ['float16', 1, [1]],
    // orig_im_size is optimized out for this model
    //'orig_im_size': ['float16', [1200., 1800.], [2]],
  },
}

function getFeeds(session, modelName) {
  let feeds = {};
  let inputs = models[modelName];
  let inputNames = session.inputNames;
  let seqlen = 128;
  let enc_seqlen = 128;

  if (['bart-large', 'bart-large-12'].indexOf(inputs) >= 0) {
    const kvdim = (modelName === 'bart-large') ? 16 : 12;
    const hiddendim = (modelName === 'bart-large') ? 1024 : 768;
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values')) {
        feeds[v] = getTensor('float32', 1., [1, kvdim, seqlen, 64]);
      }
      if (v.startsWith('encoder_attention_mask')) {
        feeds['encoder_attention_mask'] = getTensor('int64', 1n, [1, enc_seqlen]);
      }
    }
    feeds['use_cache_branch'] = getTensor('bool', false);
    feeds['input_ids'] = getTensor('int64', 99n, [1, seqlen]);
    feeds['encoder_hidden_states'] = getTensor('float32', 1, [1, enc_seqlen, hiddendim]);
  }

  if (['bert', 'bert64'].indexOf(inputs) >= 0) {
    if ([].indexOf(modelName) >= 0) {
      seqlen = 1;
    }
    const dtype = inputs == 'bert' ? 'int32' : 'int64';
    const value = inputs == 'bert' ? 99 : 99n;
    const one = inputs == 'bert' ? 1 : 1n;

    for (var k in inputNames) {
      const v = inputNames[k];
      if (v === 'input_ids') {
        feeds[v] = getTensor(dtype, value, [1, seqlen]);
      }
      if (v === 'input_mask' || v === 'attention_mask') {
        feeds[v] = getTensor(dtype, one, [1, seqlen]);
      }
      if (v === 'token_type_ids' || v == 'segment_ids') {
        feeds[v] = getTensor(dtype, one, [1, seqlen]);
      }
    }
  }

  if (inputs === 'clip') {
    feeds['input_ids'] = getTensor('int64', 49407n, [1, 77]);
    feeds['pixel_values'] = getTensor('float32', 99, [1, 3, 224, 224]);
    feeds['attention_mask'] = getTensor('int64', 1n, [1, 77]);
  }

  if (inputs === 'img224') {
    feeds[inputNames[0]] = getTensor('float32', 'random', [1, 3, 224, 224]);
  }

  if (inputs == 'llm-decoder') {
    if (modelName === 'gpt2') {
      seqlen = 8;
    } else if (modelName === 'distilgpt2') {
      seqlen = 16;
    }
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values')) {
        feeds[v] = getTensor('float32', 1., [1, 12, seqlen, 64]);
      }
    }
    feeds['use_cache_branch'] = getTensor('bool', false);
    feeds['input_ids'] = getTensor('int64', 99n, [1, seqlen]);
    feeds['attention_mask'] = getTensor('int64', 1n, [1, seqlen]);
  }


  if (inputs == 'sam-decoder') {
    feeds['image_embeddings'] = getTensor('float32', 0.5, [1, 256, 64, 64]);
    feeds['point_coords'] = new ort.Tensor(new Float32Array([327.1111, 426.875, 241.77777, 341.5, 398.22223, 498.02084]), [1, 3, 2]);
    feeds['point_labels'] = new ort.Tensor(new Float32Array([0., 2., 3.]), [1, 3]);
    feeds['mask_input'] = getTensor('float32', 0., [1, 1, 256, 256]);
    feeds['has_mask_input'] = getTensor('float32', 1., [1]);
    if (inputNames.includes('orig_im_size')) {
      feeds['orig_im_size'] = new ort.Tensor(new Float32Array([512., 512.]), [2]);
    }
  }

  if (['t5-decoder', 'flan-t5-decoder'].indexOf(inputs) >= 0) {
    seqlen = 1;
    feeds['input_ids'] = getTensor('int64', 99n, [1, seqlen]);
    feeds['encoder_hidden_states'] = getTensor('float32', 1, [1, seqlen, 512]);
    const encoder_shape = (inputs == 't5-decoder') ? [1, 8, enc_seqlen, 64] : [1, 6, enc_seqlen, 64];
    const decoder_shape = (inputs == 't5-decoder') ? [1, 8, seqlen, 64] : [1, 6, seqlen, 64];
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values.')) {
        if (v.includes('decoder')) {
          feeds[v] = getTensor('float32', 1, decoder_shape);
        } else if (v.includes('encoder')) {
          feeds[v] = getTensor('float32', 1, encoder_shape);
        }
      }
      if (v == 'encoder_attention_mask') {
        feeds['encoder_attention_mask'] = getTensor('int64', 1n, [1, enc_seqlen]);
      }
    }
    feeds['use_cache_branch'] = getTensor('bool', true);
  }

  if (inputs === 't5-encoder') {
    feeds['input_ids'] = getTensor('int64', 99n, [1, seqlen]);
  }

  if (inputs === 'whisper-decoder') {
    feeds['input_ids'] = getTensor('int64', 1n, [1, 1]);
    feeds['encoder_hidden_states'] = getTensor('float32', 'random', [1, 1500, 384]);
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values.')) {
        if (v.includes('decoder')) {
          feeds[v] = getTensor('float32', 1, [1, 6, seqlen, 64]);
        } else if (v.includes('encoder')) {
          feeds[v] = getTensor('float32', 1, [1, 6, 1500, 64]);
        }
      }
    }
    feeds['use_cache_branch'] = getTensor('bool', false);
  }

  if (isDict(inputs)) {
    for (let key in inputs) {
      let value = inputs[key];
      feeds[key] = getTensor(value[0], value[1], value[2]);
    }
  }

  return feeds;
}

function getTensor(type, data, dims) {
  let typedArray;
  if (type === 'bool') {
    return new ort.Tensor(type, [data], [1]);
  } else if (type === 'uint16') {
    typedArray = Uint16Array;
  } else if (type === 'float16') {
    typedArray = Uint16Array;
  } else if (type === 'float32') {
    typedArray = Float32Array;
  } else if (type === 'int32') {
    typedArray = Int32Array;
  } else if (type === 'int64') {
    typedArray = BigInt64Array;
  }

  let _data;
  if (Array.isArray(data)) {
    _data = data === 'random' ? Math.random() : data;
  } else {
    let size = 1;
    dims.forEach((dim) => {
      size *= dim;
    });
    _data = typedArray.from({ length: size }, () => data === 'random' ? Math.random() : data);
  }
  return new ort.Tensor(type, _data, dims);
}

function isDict(v) {
  return typeof v === 'object' && v !== null && !(v instanceof Array) && !(v instanceof Date);
}
