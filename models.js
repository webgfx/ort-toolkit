const models = {
  // daily test
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

  'mobilenetv2-12': 'img224', // from teams
  'resnet50-v2-7': 'img224', // webnn

  /*
  https://github.com/vietanhdev/samexporter
  python -m samexporter.export_decoder --checkpoint models/sam_vit_b_01ec64.pth --output models/sam-b-decoder.onnx --model-type vit_b --return-single-mask
  python -m samexporter.export_encoder --checkpoint models/sam_vit_b_01ec64.pth --output models/sam-b-encoder.onnx --model-type vit_b --use-preprocess
  python -m samexporter.export_decoder --checkpoint models/sam_vit_l_0b3195.pth --output models/sam-l-decoder.onnx --model-type vit_l --return-single-mask
  python -m samexporter.export_encoder --checkpoint models/sam_vit_l_0b3195.pth --output models/sam-l-encoder.onnx --model-type vit_l --use-preprocess
  python -m samexporter.export_decoder --checkpoint models/sam_vit_h_4b8939.pth --output models/sam-h-decoder.onnx --model-type vit_h --return-single-mask
  python -m samexporter.export_encoder --checkpoint models/sam_vit_h_4b8939.pth --output models/sam-h-encoder.onnx --model-type vit_h --use-preprocess
  */
  'sam-b-decoder': 'sam-decoder', // TODO: conformance fails

  'sd-vae-decoder': 'sd-vae-decoder',
  'sd-vae-decoder-arthur': 'sd-vae-decoder', // https://huggingface.co/aislamov/stable-diffusion-2-1-base-onnx/tree/9f697c96d42e5c09437ff14b0a2b287366ce488d/vae_decoder
  'sd-vae-decoder-f16': 'sd-vae-decoder-f16',
  'sd-vae-encoder': 'sd-vae-encoder',


  't5-small-decoder': 't5-decoder', // tjs/t5-small/onnx/decoder_model_merged.onnx
  't5-small-encoder': 't5-encoder', // tjs/t5-small/onnx/encoder_model.onnx

  'tinyyolov2-8': { 'image': ['float32', 'random', [1, 3, 416, 416]] }, // webnn
  'whisper-tiny-decoder': 'whisper-decoder', // tjs/openai/whisper-tiny/onnx/decoder_model_merged.onnx
  'whisper-tiny-encoder': { 'input_features': ['float32', 'random', [1, 80, 3000]] }, // tjs/openai/whisper-tiny/onnx/encoder_model.onnx


  // TODO
  'sam-b-encoder': 'sam-encoder', // shader issue

  'm2m100-decoder': 'm2m100-decoder', // https://huggingface.co/Xenova/m2m100/resolve/main/onnx/decoder_model_merged.onnx. RuntimeError: Aborted()
  'm2m100-encoder': 'm2m100-encoder',// https://huggingface.co/Xenova/m2m100_418M/resolve/main/onnx/encoder_model.onnx. RangeError: offset is out of bounds

  // sd-unet: Stable-Diffusion-v1.5-unet-fixed-size-batch-1-float16-no-shape-ops-embedded-weights from WebNN
  // sd-vae-decoder-f16: sd2.1-inpainting-vae-decoder-float16-zeroed-weights from WebNN
  // the rests: http://powerbuilder.sh.intel.com/project/webnn/model/w3c/stable-diffusion-v1-5/
  'sd-text-encoder': 'sd-text-encoder', // Failed to run JSEP kernel
  'sd-unet-f16': 'sd-unet-f16', // RangeError: offset is out of bounds


  // Obsolete
  'mobilenetv2-7': 'img224',
  'mobilenetv2-10': 'img224',
  'resnet50-v1-12': 'img224',
}

function getFeeds(session, modelName) {
  let feeds = {};
  let inputs = models[modelName];
  let inputNames = session.inputNames;
  let decSeqLen = 128;
  let encSeqLen = 128;

  if (['bart-large', 'bart-large-12'].indexOf(inputs) >= 0) {
    const kvdim = (modelName === 'bart-large') ? 16 : 12;
    const hiddendim = (modelName === 'bart-large') ? 1024 : 768;
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values')) {
        feeds[v] = getTensor('float32', 1., [1, kvdim, decSeqLen, 64]);
      }
      if (v.startsWith('encoder_attention_mask')) {
        feeds['encoder_attention_mask'] = getTensor('int64', 1n, [1, encSeqLen]);
      }
    }
    feeds['use_cache_branch'] = getTensor('bool', false);
    feeds['input_ids'] = getTensor('int64', 99n, [1, decSeqLen]);
    feeds['encoder_hidden_states'] = getTensor('float32', 1, [1, encSeqLen, hiddendim]);
  }

  if (['bert', 'bert64'].indexOf(inputs) >= 0) {
    if ([].indexOf(modelName) >= 0) {
      decSeqLen = 1;
    }
    const dtype = inputs == 'bert' ? 'int32' : 'int64';
    const value = inputs == 'bert' ? 99 : 99n;
    const one = inputs == 'bert' ? 1 : 1n;

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

  if (inputs === 'img224') {
    feeds[inputNames[0]] = getTensor('float32', 'random', [1, 3, 224, 224]);
  }

  if (inputs == 'llm-decoder') {
    if (modelName === 'gpt2') {
      decSeqLen = 8;
    } else if (modelName === 'distilgpt2') {
      decSeqLen = 16;
    }
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values')) {
        feeds[v] = getTensor('float32', 1., [1, 12, decSeqLen, 64]);
      }
    }
    feeds['use_cache_branch'] = getTensor('bool', false);
    feeds['input_ids'] = getTensor('int64', 99n, [1, decSeqLen]);
    feeds['attention_mask'] = getTensor('int64', 1n, [1, decSeqLen]);
  }

  if (inputs == 'm2m100-decoder') {
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

  if (inputs == 'm2m100-encoder') {
    feeds['input_ids'] = getTensor('int64', 99n, [1, encSeqLen]);
    feeds['attention_mask'] = getTensor('int64', 1n, [1, encSeqLen]);
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

  if (inputs == 'sam-encoder') {
    feeds['input_image'] = getTensor('float32', 1., [224, 224, 3]);
  }

  if (inputs == 'sd-text-encoder') {
    feeds['input_ids'] = getTensor('int32', 99, [1, encSeqLen]);
  }

  if (inputs == 'sd-unet-f16') {
    feeds['sample'] = getTensor('float16', 1, [1, 4, 64, 64]);
    feeds['timestep'] = getTensor('int64', 1n, [1]);
    feeds['encoder_hidden_states'] = getTensor('float16', 1, [1, 77, 768]);
  }

  if (inputs == 'sd-vae-decoder-f16') {
    feeds['latent_sample'] = getTensor('float16', 'random', [1, 4, 64, 64]);
  }

  if (inputs == 'sd-vae-decoder') {
    feeds['latent_sample'] = getTensor('float32', 'random', [1, 4, 64, 64]);
  }

  if (inputs == 'sd-vae-encoder') {
    feeds['sample'] = getTensor('float32', 'random', [1, 3, 512, 512]);
  }

  if (['t5-decoder', 'flan-t5-decoder'].indexOf(inputs) >= 0) {
    decSeqLen = 1;
    feeds['input_ids'] = getTensor('int64', 99n, [1, decSeqLen]);
    feeds['encoder_hidden_states'] = getTensor('float32', 1, [1, decSeqLen, 512]);
    const encoder_shape = (inputs == 't5-decoder') ? [1, 8, encSeqLen, 64] : [1, 6, encSeqLen, 64];
    const decoder_shape = (inputs == 't5-decoder') ? [1, 8, decSeqLen, 64] : [1, 6, decSeqLen, 64];
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
    feeds['use_cache_branch'] = getTensor('bool', true);
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
    _data = data;
  } else {
    let size = 1;
    dims.forEach((dim) => {
      size *= dim;
    });
    if (data === 'ramdom') {
      _data = typedArray.from({ length: size }, () => Math.random());
    } else if (data === 'ramp') {
      _data = typedArray.from({ length: size }, (_, i) => i);
    } else {
      _data = typedArray.from({ length: size }, () => data);
    }

  }
  return new ort.Tensor(type, _data, dims);
}

function isDict(v) {
  return typeof v === 'object' && v !== null && !(v instanceof Array) && !(v instanceof Date);
}
