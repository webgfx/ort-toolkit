const models = {
  // daily test
  // tjs/albert-base-v2/onnx/model.onnx. TODO: NaN
  'albert-base-v2': ['bert64', {batch_size: 1, sequence_length: 128}],
  // https://huggingface.co/Xenova/albert-base-v2/tree/main/onnx
  'albert-base-v2-i8': 'bert64',
  // tjs/facebook/bart-large-cnn/onnx/encoder_model.onnx
  'bart-large-cnn-encoder': ['bert64', {batch_size: 1, encoder_sequence_length: 128}],
  // tjs/bert-base-cased/onnx/model.onnx
  'bert-base-cased': ['bert64', {batch_size: 1, sequence_length: 9}],
  // tjs/bert-base-uncased/onnx/model.onnx
  'bert-base-uncased': ['bert64', {batch_size: 1, sequence_length: 128}],
  // tjs/openai/clip-vit-base-patch16/onnx/model.onnx
  'clip-vit-base-patch16': [
    'clip',
    {text_batch_size: 1, sequence_length: 77, image_batch_size: 1, num_channels: 3, height: 224, width: 224},
  ],
  // webnn
  'densenet-9': 'img224',
  // tjs/facebook/detr-resnet-50/onnx/model.onnx. TODO: conformance fails
  'detr-resnet-50': ['img224', {batch_size: 1, num_channels: 3, height: 224, width: 224}],
  // https://huggingface.co/Xenova/detr-resnet-50/tree/main/onnx/model.onnx
  'detr-resnet-50-2': 'detr-resnet-50-2',
  // tjs/facebook/dino-vitb16/onnx/model.onnx
  'dino-vitb16': ['img224', {batch_size: 1, num_channels: 3, height: 224, width: 224}],
  // tjs/distilbert-base-uncased/onnx/model.onnx
  'distilbert-base-uncased': ['bert64', {batch_size: 1, sequence_length: 50}],
  // https://huggingface.co/Xenova/distilgpt2/blob/main/onnx/decoder_model.onnx. TODO: NaN
  'distilgpt2-decoder': ['llm-decoder', {batch_size: 1, sequence_length: 16}],
  // https://huggingface.co/Xenova/distilgpt2/blob/main/onnx/decoder_model_merged.onnx. TODO: freeDimensionOverrides
  // {attention_mask_sequence_length: 16, batch_size: 1, past_sequence_length: 64, sequence_length: 16}
  'distilgpt2-decoder-merged': ['llm-decoder'],
  // webnn
  'efficientnet-lite4-11': {'images:0': ['float32', 'random', [1, 224, 224, 3]]},
  // webnn
  'emotion-ferplus-8': {Input3: ['float32', 'random', [1, 1, 64, 64]]},
  // https://huggingface.co/gpt2/blob/main/onnx/decoder_model.onnx. TODO: NaN
  'gpt2-decoder': ['llm-decoder', {batch_size: 1, sequence_length: 8}],
  // https://huggingface.co/gpt2/blob/main/onnx/decoder_model_merged.onnx. TODO: freeDimensionOverrides
  // {attention_mask_sequence_length: 16, batch_size: 1, past_sequence_length: 16, sequence_length: 8}
  'gpt2-decoder-merged': ['llm-decoder'],
  // https://huggingface.co/Xenova/m2m100_418M/resolve/main/onnx/encoder_model.onnx
  'm2m100-encoder': ['m2m100-encoder', {batch_size: 1, encoder_sequence_length: 128}],
  // from teams
  'mobilenetv2-12': ['img224', {batch_size: 1}],
  // https://huggingface.co/webml/models/tree/main
  // not sure if its really 12
  'mobilenetv2-12-f16': 'img224-f16',
  // https://huggingface.co/webml/models/tree/main
  'mobilenetv2-12-i8': 'img224',

  'mobilenetv3': 'mobilenetv3',

  // https://huggingface.co/webml/models/tree/main
  'realesrgan-t1024': 'realesrgan',
  'realesrgan-t512': 'realesrgan',
  'realesrgan-t256': 'realesrgan',
  'realesrgan-t128': 'realesrgan',
  'realesrgan-t64': 'realesrgan',
  'realesrgan-t1024-f16': 'realesrgan',
  'realesrgan-t512-f16': 'realesrgan',
  'realesrgan-t256-f16': 'realesrgan',
  'realesrgan-t128-f16': 'realesrgan',
  'realesrgan-t64-f16': 'realesrgan',

  // webnn
  'resnet50-v2-7': ['img224', {N: 1}],

  /*
      https://github.com/vietanhdev/samexporter
      python -m samexporter.export_decoder --checkpoint models/sam_vit_b_01ec64.pth --output models/sam-b-decoder.onnx
      --model-type vit_b --return-single-mask python -m samexporter.export_encoder --checkpoint
      models/sam_vit_b_01ec64.pth --output models/sam-b-encoder.onnx --model-type vit_b --use-preprocess

      python -m samexporter.export_decoder --checkpoint models/sam_vit_l_0b3195.pth --output models/sam-l-decoder.onnx
      --model-type vit_l --return-single-mask python -m samexporter.export_encoder --checkpoint
      models/sam_vit_l_0b3195.pth --output models/sam-l-encoder.onnx --model-type vit_l --use-preprocess

      python -m samexporter.export_decoder --checkpoint models/sam_vit_h_4b8939.pth --output models/sam-h-decoder.onnx
      --model-type vit_h --return-single-mask python -m samexporter.export_encoder --checkpoint
      models/sam_vit_h_4b8939.pth --output models/sam-h-encoder.onnx --model-type vit_h --use-preprocess
      */
  'sam-b-decoder': ['sam-decoder', {num_points: 2}],  // TODO: conformance fails
  'sam-b-encoder': ['sam-encoder', {image_height: 224, image_width: 224}],
  // https://huggingface.co/webml/models/blob/main/fp16/segment-anything-vit-h-static-shapes-origin-im-size-initializer-optimized-float16.onnx
  'sam-h-decoder-f16': 'sam-decoder-f16',

  'sd15-vae-decoder': ['sd-vae-decoder', {batch: 1, channels: 4, height: 64, width: 64}],
  'sd15-vae-encoder': ['sd-vae-encoder', {batch: 1, channels: 3, height: 512, width: 512}],

  'sd21-vae-decoder-f16': ['sd-vae-decoder-f16', {batch: 1, channels: 4, height: 64, width: 64}],
  'sd21-vae-encoder': [
    'sd-vae-encoder',
    {vaeenc_sample_batch: 1, vaeenc_sample_channels: 3, vaeenc_sample_height: 512, vaeenc_sample_width: 512}
  ],

  // https://huggingface.co/Xenova/t5-small/blob/main/onnx/decoder_model.onnx
  't5-small-decoder': ['t5-decoder', {batch_size: 1, decoder_sequence_length: 128, encoder_sequence_length: 128}],
  // https://huggingface.co/Xenova/t5-small/blob/main/onnx/decoder_model_merged.onnx. TODO: freeDimensionOverrides
  /*
  {
    batch_size: 1, decoder_sequence_length: 128, encoder_sequence_length: 128, encoder_sequence_length_out: 16,
        past_decoder_sequence_length: 16
  }
  */
  't5-small-decoder-merged': ['t5-decoder'],
  // tjs/t5-small/onnx/encoder_model.onnx
  't5-small-encoder': ['t5-encoder', {batch: 1, sequence: 128}],

  // webnn
  'tinyyolov2-8': [{image: ['float32', 'random', [1, 3, 416, 416]]}, {None: 1}],
  // https://huggingface.co/Xenova/whisper-tiny/blob/main/onnx/decoder_model.onnx
  'whisper-tiny-decoder': ['whisper-decoder', 'whisper-decoder'],
  // https://huggingface.co/Xenova/whisper-tiny/blob/main/onnx/decoder_model_merged.onnx
  'whisper-tiny-decoder-merged': ['whisper-decoder'],
  // https://huggingface.co/Xenova/whisper-tiny/blob/main/onnx/encoder_model.onnx
  'whisper-tiny-encoder': [
    {input_features: ['float32', 'random', [1, 80, 3000]]},
    {batch_size: 1, feature_size: 80, encoder_sequence_length: 3000}
  ],

  // TODO
  // https://huggingface.co/Xenova/m2m100/resolve/main/onnx/decoder_model_merged.onnx
  // RuntimeError: Aborted()
  'm2m100-decoder': 'm2m100-decoder',

  // sd-unet: Stable-Diffusion-v1.5-unet-fixed-size-batch-1-float16-no-shape-ops-embedded-weights from WebNN.
  // The rests: http://powerbuilder.sh.intel.com/project/webnn/model/w3c/stable-diffusion-v1-5/
  'sd15-text-encoder': 'sd-text-encoder',           // Failed to run JSEP kernel
  'sd15-unet-f16': ['sd-unet-f16', 'sd-unet-f16'],  // RangeError: offset is out of bounds

  // https://huggingface.co/aislamov/stable-diffusion-2-1-base-onnx/blob/main/
  // sd-vae-decoder-f16: sd2.1-inpainting-vae-decoder-float16-zeroed-weights from WebNN.
  'sd21-text-encoder': 'sd-text-encoder',
  'sd21-unet': 'sd-unet',

  // Deprecated
  // webnn. If the value is set to 0.5, conformance test would fail.
  // op unsample is deprecated (https://github.com/onnx/onnx/blob/main/docs/Operators.md#Upsample), so we move this
  // to deprecated
  'candy-8': 'img224',

  'mobilenetv2-7': ['img224', {batch_size: 1}],
  'mobilenetv2-10': ['img224', {batch_size: 1}],
  'resnet50-v1-12': ['img224', {N: 1}],
  // https://huggingface.co/aislamov/stable-diffusion-2-1-base-onnx/tree/9f697c96d42e5c09437ff14b0a2b287366ce488d/vae_decoder
  'sd-vae-decoder-arthur': 'sd-vae-decoder',

  // Temp
  'sam-b-vision-encoder': 'sam-b-vision-encoder',
};

function getFeeds(session, modelName) {
  let feeds = {};
  let inputs = models[modelName];
  if (inputs instanceof Array) {
    inputs = inputs[0];
  }
  let inputNames = session.inputNames;
  let decSeqLen = 128;
  let encSeqLen = 128;

  if (['bart-large', 'bart-large-12'].indexOf(inputs) >= 0) {
    const kvdim = modelName === 'bart-large' ? 16 : 12;
    const hiddendim = modelName === 'bart-large' ? 1024 : 768;
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values')) {
        feeds[v] = getTensor('float32', 1, [1, kvdim, decSeqLen, 64]);
      }
      if (v.startsWith('encoder_attention_mask')) {
        feeds['encoder_attention_mask'] = getTensor('int64', 1n, [1, encSeqLen]);
      }
    }
    feeds['use_cache_branch'] = getTensor('bool', true);
    feeds['input_ids'] = getTensor('int64', 99n, [1, decSeqLen]);
    feeds['encoder_hidden_states'] = getTensor('float32', 1, [1, encSeqLen, hiddendim]);
  }

  if (['bert', 'bert64'].indexOf(inputs) >= 0) {
    if (modelName === 'bert-base-cased') {
      decSeqLen = 9;
    } else if (modelName === 'distilbert-base-uncased') {
      decSeqLen = 50;
    }
    const dtype = inputs === 'bert' ? 'int32' : 'int64';
    const value = inputs === 'bert' ? 99 : 99n;
    const one = inputs === 'bert' ? 1 : 1n;

    for (var k in inputNames) {
      const v = inputNames[k];
      if (v === 'input_ids') {
        feeds[v] = getTensor(dtype, value, [1, decSeqLen]);
      }
      if (v === 'input_mask' || v === 'attention_mask') {
        feeds[v] = getTensor(dtype, one, [1, decSeqLen]);
      }
      if (v === 'token_type_ids' || v == 'segment_ids') {
        feeds[v] = getTensor(dtype, one, [1, decSeqLen]);
      }
    }
  }

  if (inputs === 'clip') {
    feeds['input_ids'] = getTensor('int64', 49407n, [1, 77]);
    feeds['pixel_values'] = getTensor('float32', 99, [1, 3, 224, 224]);
    feeds['attention_mask'] = getTensor('int64', 1n, [1, 77]);
  }

  if (inputs === 'detr-resnet-50-2') {
    feeds['pixel_values'] = getTensor('float32', 'random', [1, 3, 800, 800]);
    feeds['pixel_mask'] = getTensor('int64', 1n, [1, 64, 64]);
  }

  if (inputs === 'img224') {
    feeds[inputNames[0]] = getTensor('float32', 'random', [1, 3, 224, 224]);
  }

  if (inputs === 'img224-f16') {
    feeds[inputNames[0]] = getTensor('float16', 'random', [1, 3, 224, 224]);
  }

  if (inputs === 'img224-i8') {
    feeds[inputNames[0]] = getTensor('int8', 'random', [1, 3, 224, 224]);
  }

  if (inputs === 'llm-decoder') {
    if (modelName === 'gpt2-decoder') {
      decSeqLen = 8;
    } else if (['distilgpt2-decoder', 'distilgpt2-decoder-merged'].indexOf(modelName) >= 0) {
      decSeqLen = 16;
    }
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values')) {
        feeds[v] = getTensor('float32', 1, [1, 12, decSeqLen, 64]);
      }
    }
    feeds['input_ids'] = getTensor('int64', 99n, [1, decSeqLen]);
    feeds['attention_mask'] = getTensor('int64', 1n, [1, decSeqLen]);

    if (modelName.endsWith('merged')) {
      feeds['use_cache_branch'] = getTensor('bool', true);
    }
  }

  if (inputs === 'm2m100-decoder') {
    feeds['encoder_attention_mask'] = getTensor('int64', 1n, [1, encSeqLen]);
    feeds['input_ids'] = getTensor('int64', 99n, [1, decSeqLen]);
    feeds['encoder_hidden_states'] = getTensor('float32', 1, [1, encSeqLen, 1024]);
    const encoder_shape = [1, 16, encSeqLen, 64];
    const decoder_shape = [1, 16, decSeqLen, 64];
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values.')) {
        if (v.includes('decoder')) {
          feeds[v] = getTensor('float32', 1, decoder_shape);
        } else if (v.includes('encoder')) {
          feeds[v] = getTensor('float32', 1, encoder_shape);
        }
      }
    }
    feeds['use_cache_branch'] = getTensor('bool', true);
  }

  if (inputs === 'm2m100-encoder') {
    feeds['input_ids'] = getTensor('int64', 99n, [1, encSeqLen]);
    feeds['attention_mask'] = getTensor('int64', 1n, [1, encSeqLen]);
  }

  if (inputs === 'mobilenetv3') {
    feeds[inputNames[0]] = getTensor('float32', 'random', [1, 224, 224, 3]);
  }

  if (inputs === 'realesrgan') {
    const modelInfo = modelName.split('-');
    const tileSize = parseInt(modelInfo[1].replace('t', ''));
    const dataType = modelName.endsWith('f16') ? '16' : '32';
    feeds[`in_image_float${dataType}_rgb01`] = getTensor(`float${dataType}`, 'random', [1, 3, tileSize, tileSize]);
  }

  if (inputs === 'sam-b-vision-encoder') {
    feeds['pixel_values'] = getTensor('float32', 'random', [1, 3, 1024, 1024]);
  }

  if (inputs === 'sam-decoder') {
    feeds['image_embeddings'] = getTensor('float32', 'random', [1, 256, 64, 64]);
    feeds['point_coords'] = getTensor('float32', 'random', [1, 2, 2]);
    feeds['point_labels'] = getTensor('float32', 'random', [1, 2]);
    feeds['mask_input'] = getTensor('float32', 'random', [1, 1, 256, 256]);
    feeds['has_mask_input'] = getTensor('float32', 'random', [1]);
    if (inputNames.includes('orig_im_size')) {
      feeds['orig_im_size'] = new ort.Tensor(new Float32Array([512, 512]), [2]);
    }
  }

  if (inputs === 'sam-decoder-f16') {
    feeds['image_embeddings'] = getTensor('float16', 'random', [1, 256, 64, 64]);
    feeds['point_coords'] = getTensor('float16', 'random', [1, 2, 2]);
    feeds['point_labels'] = getTensor('float16', 'random', [1, 2]);
    feeds['mask_input'] = getTensor('float16', 'random', [1, 1, 256, 256]);
    feeds['has_mask_input'] = getTensor('float16', 'random', [1]);
    if (inputNames.includes('orig_im_size')) {
      feeds['orig_im_size'] = new ort.Tensor(new Float32Array([512, 512]), [2]);
    }
  }

  if (inputs === 'sam-encoder') {
    feeds['input_image'] = getTensor('float32', 1, [224, 224, 3]);
  }

  if (inputs === 'sd-text-encoder') {
    feeds['input_ids'] = getTensor('int32', 99, [1, encSeqLen]);
  }

  if (inputs === 'sd-unet') {
    feeds['sample'] = getTensor('float32', 1, [1, 4, 64, 64]);
    feeds['timestep'] = getTensor('int64', 1n, [1]);
    feeds['encoder_hidden_states'] = getTensor('float32', 1, [1, 77, 768]);
  }

  if (inputs === 'sd-unet-f16') {
    feeds['sample'] = getTensor('float16', 1, [1, 4, 64, 64]);
    feeds['timestep'] = getTensor('int64', 1n, [1]);
    feeds['encoder_hidden_states'] = getTensor('float16', 1, [1, 77, 768]);
  }

  if (inputs === 'sd-vae-decoder-f16') {
    feeds['latent_sample'] = getTensor('float16', 'random', [1, 4, 64, 64]);
  }

  if (inputs === 'sd-vae-decoder') {
    feeds['latent_sample'] = getTensor('float32', 'random', [1, 4, 64, 64]);
  }

  if (inputs === 'sd-vae-encoder') {
    feeds['sample'] = getTensor('float32', 'random', [1, 3, 512, 512]);
  }

  if (inputs === 't5-decoder') {
    decSeqLen = 128;
    feeds['input_ids'] = getTensor('int64', 99n, [1, decSeqLen]);
    feeds['encoder_hidden_states'] = getTensor('float32', 1, [1, encSeqLen, 512]);
    const encoder_shape = inputs === 't5-decoder' ? [1, 8, encSeqLen, 64] : [1, 6, encSeqLen, 64];
    const decoder_shape = inputs === 't5-decoder' ? [1, 8, decSeqLen, 64] : [1, 6, decSeqLen, 64];
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
        feeds['encoder_attention_mask'] = getTensor('int64', 1n, [1, encSeqLen]);
      }
    }
    if (modelName.endsWith('merged')) {
      feeds['use_cache_branch'] = getTensor('bool', true);
    }
  }

  if (inputs === 't5-encoder') {
    feeds['input_ids'] = getTensor('int64', 99n, [1, decSeqLen]);
  }

  if (inputs === 'whisper-decoder') {
    feeds['input_ids'] = getTensor('int64', 1n, [1, 1]);
    feeds['encoder_hidden_states'] = getTensor('float32', 'random', [1, 1500, 384]);
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values.')) {
        if (v.includes('decoder')) {
          feeds[v] = getTensor('float32', 1, [1, 6, decSeqLen, 64]);
        } else if (v.includes('encoder')) {
          feeds[v] = getTensor('float32', 1, [1, 6, 1500, 64]);
        }
      }
    }
    if (modelName.endsWith('merged')) {
      feeds['use_cache_branch'] = getTensor('bool', true);
    }
  }

  if (isDict(inputs)) {
    for (let key in inputs) {
      let value = inputs[key];
      feeds[key] = getTensor(value[0], value[1], value[2]);
    }
  }

  return feeds;
}

function getFreeDimensionOverrides(modelName) {
  const modelInfo = models[modelName];
  if (!(modelInfo instanceof Array) || modelInfo.length == 1) {
    return null;
  }

  let freeDimensionOverrides = models[modelName][1];
  if (freeDimensionOverrides === 'sd-unet-f16') {
    freeDimensionOverrides = {
      unet_time_batch: 1,
      unet_sample_channels: 4,
      unet_sample_height: 64,
      unet_sample_width: 64,
      unet_hidden_batch: 1,
      unet_hidden_sequence: 77,
    };
  } else if (freeDimensionOverrides == 'whisper-decoder') {
    freeDimensionOverrides = {
      batch_size: 1,
      decoder_sequence_length: 1,
      past_decoder_sequence_length: 128,
      encoder_sequence_length_out: 1500,
      encoder_sequence_length: 3000,
    };
  }

  return freeDimensionOverrides;
}

function getTensor(type, data, dims) {
  let typedArray;
  if (type === 'bool') {
    return new ort.Tensor(type, [data], [1]);
  } else if (type === 'int8') {
    typedArray = Int8Array;
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
  if (Array.isArray(data) || ArrayBuffer.isView(data)) {
    _data = data;
  } else {
    let size = 1;
    dims.forEach((dim) => {
      size *= dim;
    });
    if (data === 'random') {
      _data = typedArray.from({length: size}, () => Math.random());
    } else if (data === 'ramp') {
      _data = typedArray.from({length: size}, (_, i) => i);
    } else {
      _data = typedArray.from({length: size}, () => data);
    }
  }
  return new ort.Tensor(type, _data, dims);
}

function isDict(v) {
  return typeof v === 'object' && v !== null && !(v instanceof Array) && !(v instanceof Date);
}
