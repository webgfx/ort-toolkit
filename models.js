const models = {
  // daily test
  // tjs/albert-base-v2/onnx/model.onnx.
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
  // https://huggingface.co/Xenova/codegen-350M-mono/tree/main/onnx
  'codegen-350m-mono-decoder': ['codegen-350m-mono-decoder', {'batch_size': 1, 'sequence_length': 8}],
  'codegen-350m-mono-decoder-merged': 'codegen-350m-mono-decoder',
  // webnn
  'densenet-9': 'img224',
  // tjs/facebook/detr-resnet-50/onnx/model.onnx. TODO: conformance fails
  'detr-resnet-50': ['img224', {batch_size: 1, num_channels: 3, height: 224, width: 224}],
  // https://huggingface.co/Xenova/detr-resnet-50/tree/main/onnx/model.onnx
  'detr-resnet-50-2': 'detr-resnet-50-2',
  // tjs/facebook/dino-vitb16/onnx/model.onnx
  'dino-vitb16': ['img224', {batch_size: 1, num_channels: 3, height: 224, width: 224}],
  // https://huggingface.co/Xenova/distilbart-cnn-6-6/blob/main/onnx/decoder_model.onnx
  'distilbart-cnn-6-6-decoder':
      ['distilbart-cnn-6-6-decoder', {'batch_size': 1, 'decoder_sequence_length': 168, 'encoder_sequence_length': 168}],
  // https://huggingface.co/Xenova/distilbart-cnn-6-6/blob/main/onnx/decoder_model_merged.onnx
  'distilbart-cnn-6-6-decoder-merged': 'distilbart-cnn-6-6-decoder',
  // https://huggingface.co/Xenova/distilbart-cnn-6-6/blob/main/onnx/encoder_model.onnx
  'distilbart-cnn-6-6-encoder': [
    {'input_ids': ['int64', 99n, [1, 168]], 'attention_mask': ['int64', 1n, [1, 168]]},
    {'batch_size': 1, 'encoder_sequence_length': 168}
  ],

  // tjs/distilbert-base-uncased/onnx/model.onnx
  'distilbert-base-uncased': ['bert64', {batch_size: 1, sequence_length: 50}],
  // https://huggingface.co/Xenova/distilgpt2/blob/main/onnx/decoder_model.onnx.
  'distilgpt2-decoder': ['llm-decoder', {batch_size: 1, sequence_length: 16}],
  // https://huggingface.co/Xenova/distilgpt2/blob/main/onnx/decoder_model_merged.onnx. TODO: freeDimensionOverrides
  // {attention_mask_sequence_length: 16, batch_size: 1, past_sequence_length: 64, sequence_length: 16}
  'distilgpt2-decoder-merged': 'llm-decoder',
  // webnn
  'efficientnet-lite4-11': {'images:0': ['float32', 'random', [1, 224, 224, 3]]},

  // https://huggingface.co/Xenova/flan-t5-small/blob/main/onnx/encoder_model.onnx
  'flan-t5-small-encoder': [
    {'input_ids': ['int64', 99n, [1, 128]], 'attention_mask': ['int64', 1n, [1, 128]]},
    {'batch_size': 1, 'encoder_sequence_length': 128}
  ],
  // https://huggingface.co/Xenova/flan-t5-small/blob/main/onnx/decoder_model.onnx
  'flan-t5-small-decoder':
      ['flan-t5-decoder', {'batch_size': 1, 'decoder_sequence_length': 128, 'encoder_sequence_length': 128}],
  // https://huggingface.co/Xenova/flan-t5-small/blob/main/onnx/decoder_model_merged.onnx
  'flan-t5-small-decoder-merged': 'flan-t5-decoder',
  // webnn
  'emotion-ferplus-8': {Input3: ['float32', 'random', [1, 1, 64, 64]]},
  // https://huggingface.co/gpt2/blob/main/onnx/decoder_model.onnx. TODO: NaN
  'gpt2-decoder': ['llm-decoder', {batch_size: 1, sequence_length: 8}],
  // https://huggingface.co/gpt2/blob/main/onnx/decoder_model_merged.onnx. TODO: freeDimensionOverrides
  // {attention_mask_sequence_length: 16, batch_size: 1, past_sequence_length: 16, sequence_length: 8}
  'gpt2-decoder-merged': 'llm-decoder',
  // https://huggingface.co/Xenova/m2m100_418M/resolve/main/onnx/encoder_model.onnx
  'm2m100-encoder': ['m2m100-encoder', {batch_size: 1, encoder_sequence_length: 128}],
  // from teams
  'mobilenetv2-12': ['img224', {batch_size: 1}],
  // https://huggingface.co/webml/models/tree/main
  // not sure if its really 12
  'mobilenetv2-12-f16': 'img224-f16',
  // https://huggingface.co/webml/models/tree/main
  'mobilenetv2-12-i8': 'img224',

  // https://github.com/onnx/models/tree/main/Computer_Vision/mobilenetv3_small_100_Opset17_timm
  'mobilenetv3-small-100': 'img224',

  // https://huggingface.co/Xenova/mobilevit-small/blob/main/onnx/model.onnx
  'mobilevit-small': [
    {'pixel_values': ['float32', 'random', [1, 3, 256, 256]]},
    {'batch_size': 1, 'num_channels': 3, 'height': 256, 'width': 256}
  ],

  // https://huggingface.co/Xenova/msmarco-distilbert-base-v4/blob/main/onnx/model.onnx
  'msmarco-distilbert-base-v4': [
    {'input_ids': ['int64', 99n, [1, 50]], 'attention_mask': ['int64', 1n, [1, 50]]},
    {'batch_size': 1, 'sequence_length': 50}
  ],

  // https://huggingface.co/Xenova/mt5-small/blob/main/onnx/decoder_model.onnx
  'mt5-small-decoder':
      ['mt5-decoder', {'batch_size': 1, 'decoder_sequence_length': 128, 'encoder_sequence_length': 128}],
  // https://huggingface.co/Xenova/mt5-small/blob/main/onnx/decoder_model_merged.onnx
  'mt5-small-decoder-merged': 'mt5-decoder',

  // https://huggingface.co/Xenova/mt5-small/blob/main/onnx/encoder_model.onnx
  'mt5-small-encoder': [
    {'input_ids': ['int64', 99n, [1, 128]], 'attention_mask': ['int64', 1n, [1, 128]]},
    {'batch_size': 1, 'encoder_sequence_length': 128}
  ],

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

  // https://huggingface.co/Xenova/squeezebert-uncased/blob/main/onnx/model.onnx
  'squeezebert-uncased': [
    {
      'input_ids': ['int64', 99n, [1, 50]],
      'attention_mask': ['int64', 1n, [1, 50]],
      'token_type_ids': ['int64', 99n, [1, 50]],
    },
    {'batch_size': 1, 'sequence_length': 50}
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
  't5-small-decoder-merged': 't5-decoder',
  // tjs/t5-small/onnx/encoder_model.onnx
  't5-small-encoder': ['t5-encoder', {batch: 1, sequence: 128}],

  // webnn
  'tinyyolov2-8': [{image: ['float32', 'random', [1, 3, 416, 416]]}, {None: 1}],

  // https://huggingface.co/Xenova/vit-base-patch16-224/blob/main/onnx/model.onnx
  'vit-base-patch16-224': [
    {'pixel_values': ['float32', 1, [1, 3, 224, 224]]},
    {'batch_size': 1, 'num_channels': 3, 'height': 224, 'width': 224}
  ],

  // https://huggingface.co/Xenova/vit-gpt2-image-captioning/blob/main/onnx/decoder_model.onnx
  'vit-gpt2-image-captioning-decoder': [
    'vit-gpt2-image-captioning-decoder',
    {'batch_size': 1, 'decoder_sequence_length': 168, 'encoder_sequence_length': 168}
  ],
  // https://huggingface.co/Xenova/vit-gpt2-image-captioning/blob/main/onnx/decoder_model_merged.onnx
  'vit-gpt2-image-captioning-decoder-merged': 'vit-gpt2-image-captioning-decoder',
  // https://huggingface.co/Xenova/vit-gpt2-image-captioning/blob/main/onnx/encoder_model.onnx
  'vit-gpt2-image-captioning-encoder': [
    {'pixel_values': ['float32', 1, [1, 3, 224, 224]]},
    {'batch_size': 1, 'num_channels': 3, 'height': 224, 'width': 224}
  ],

  // https://huggingface.co/Xenova/whisper-tiny/blob/main/onnx/decoder_model.onnx
  'whisper-tiny-decoder': ['whisper-decoder', 'whisper-decoder'],
  // https://huggingface.co/Xenova/whisper-tiny/blob/main/onnx/decoder_model_merged.onnx
  'whisper-tiny-decoder-merged': 'whisper-decoder',
  // https://huggingface.co/Xenova/whisper-tiny/blob/main/onnx/encoder_model.onnx
  'whisper-tiny-encoder': [
    {input_features: ['float32', 'random', [1, 80, 3000]]},
    {batch_size: 1, feature_size: 80, encoder_sequence_length: 3000}
  ],

  // https://huggingface.co/Xenova/xlm-roberta-base/blob/main/onnx/model.onnx
  'xlm-roberta-base': [
    {'input_ids': ['int64', 99n, [1, 50]], 'attention_mask': ['int64', 1n, [1, 50]]},
    {'batch_size': 1, 'sequence_length': 50}
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

const modelEpsilons = {
  'sam-h-decoder-f16': [1, 1], // TODO: Check the conformance
  'mt5-small-decoder': [0.1, 0.06],
}

function getEpsilons(modelName) {
  let epsilons;

  if (modelName.endsWith("-f16")) {
    epsilons = [0.1, 0.05];
  } else {
    epsilons = [0.1, 0.005];
  }

  if (modelName in modelEpsilons) {
    epsilons = modelEpsilons[modelName];
  }

  return epsilons;
}

function getRandomIntInclusive(min, max) {
  const minCeiled = Math.ceil(min);
  const maxFloored = Math.floor(max);
  // The maximum is inclusive and the minimum is inclusive
  return Math.floor(Math.random() * (maxFloored - minCeiled + 1) + minCeiled);
}

function getRandom(type) {
  let min, max;

  if (type === 'bool') {
    min = 0;
    max = 1;
    return getRandomIntInclusive(min, max);
  } else if (type === 'int8') {
    min = -(2 ** 7);
    max = 2 ** 7 - 1;
    return getRandomIntInclusive(min, max);
  } else if (type === 'float16') {
    // F16 valid bits range: (Positive) 0x0000~0x7BFF (Negative) 0x8000~0xFBFF
    // F16 valid numeric range: -65504~65504

    // 1.0 -> 0x3C00. Use this specific value so that results of WebGPU and WASM can be somehow compared.
    return 0x3C00;

    // min = 0;
    // max = 1000;
    // return getRandomIntInclusive(min, max);

  } else if (type === 'int32') {
    min = -(2 ** 31);
    max = 2 ** 31 - 1;
    return getRandomIntInclusive(min, max);
  } else if (type === 'uint32') {
    min = 0;
    max = 2 ** 32 - 1;
    return getRandomIntInclusive(min, max);
  } else if (type === 'float32') {
    return Math.random() * 10;
  } else if (type === 'int64') {
    min = -(2 ** 63);
    max = 2 ** 63 - 1;
    return getRandomIntInclusive(min, max);
  }
}

// depend on global variables: feedsInfo, runTimes, session, warmupTimes
function getFeedInfo(feed, type, data, dims) {
  if (!session.inputNames.includes(feed)) {
    return;
  }
  for (i = 0; i < warmupTimes + runTimes; i++) {
    let typedArray;
    let typeBytes;
    if (type === 'bool') {
      data = [data];
      dims = [1];
      typeBytes = 1;
    } else if (type === 'int8') {
      typedArray = Int8Array;
    } else if (type === 'float16') {
      typedArray = Uint16Array;
    } else if (type === 'int32') {
      typedArray = Int32Array;
    } else if (type === 'uint32') {
      typedArray = Uint32Array;
    } else if (type === 'float32') {
      typedArray = Float32Array;
    } else if (type === 'int64') {
      typedArray = BigInt64Array;
    }
    if (typeBytes === undefined) {
      typeBytes = typedArray.BYTES_PER_ELEMENT;
    }

    let size, _data;
    if (Array.isArray(data) || ArrayBuffer.isView(data)) {
      size = data.length;
      _data = data;
    } else {
      size = dims.reduce((a, b) => a * b);
      if (data === 'random') {
        _data = typedArray.from({length: size}, () => getRandom(type));
      } else {
        _data = typedArray.from({length: size}, () => data);
      }
    }

    if (i > feedsInfo.length - 1) {
      feedsInfo.push(new Map());
    }
    feedsInfo[i].set(feed, [type, _data, dims, Math.ceil(size * typeBytes / 16) * 16]);
  }
}

// depend on global variables: session
function getPastKeyValuesInfo(dims) {
  for (var inputName of session.inputNames) {
    if (inputName.startsWith('past_key_values.')) {
      getFeedInfo(inputName, 'float32', 1, dims);
    }
  }
}

function getFeedsInfo(modelName) {
  let inputs = models[modelName];
  if (inputs instanceof Array) {
    inputs = inputs[0];
  }
  const inputNames = session.inputNames;
  let decSeqLen = 128;
  let encSeqLen = 128;
  let batchSize = 1;
  let seqLen = 128;

  if (['bart-large', 'bart-large-12'].indexOf(inputs) >= 0) {
    const kvDim = modelName === 'bart-large' ? 16 : 12;
    const hiddenDim = modelName === 'bart-large' ? 1024 : 768;

    getFeedInfo('encoder_attention_mask', 'int64', 1n, [1, encSeqLen]);
    getFeedInfo('input_ids', 'int64', 99n, [1, decSeqLen]);
    getFeedInfo('encoder_hidden_states', 'float32', 1, [1, encSeqLen, hiddenDim]);
    getPastKeyValuesInfo([1, kvDim, seqLen, 64]);
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

    getFeedInfo('input_ids', dtype, value, [1, decSeqLen]);
    getFeedInfo('input_mask', dtype, one, [1, decSeqLen]);
    getFeedInfo('attention_mask', dtype, one, [1, decSeqLen]);
    getFeedInfo('token_type_ids', dtype, one, [1, decSeqLen]);
    getFeedInfo('segment_ids', dtype, one, [1, decSeqLen]);
  }

  if (inputs === 'clip') {
    getFeedInfo('input_ids', 'int64', 49407n, [1, 77]);
    getFeedInfo('pixel_values', 'float32', 99, [1, 3, 224, 224]);
    getFeedInfo('attention_mask', 'int64', 1n, [1, 77]);
  }

  if (inputs === 'codegen-350m-mono-decoder') {
    getFeedInfo('attention_mask', 'int64', 1n, [1, 8]);
    getFeedInfo('input_ids', 'int64', 99n, [1, 8]);
    getPastKeyValuesInfo([batchSize, 16, seqLen, 64]);
  }

  if (inputs === 'detr-resnet-50-2') {
    getFeedInfo('pixel_values', 'float32', 'random', [1, 3, 800, 800]);
    getFeedInfo('pixel_mask', 'int64', 1n, [1, 64, 64]);
  }

  if (inputs === 'distilbart-cnn-6-6-decoder') {
    decSeqLen = 168;
    encSeqLen = 168;
    getFeedInfo('input_ids', 'int64', 99n, [1, decSeqLen]);
    getFeedInfo('encoder_attention_mask', 'int64', 1n, [1, encSeqLen]);
    getFeedInfo('encoder_hidden_states', 'float32', 1, [1, encSeqLen, 1024]);
    getPastKeyValuesInfo([1, 16, seqLen, 64]);
  }

  if (inputs === 'img224') {
    getFeedInfo(inputNames[0], 'float32', 'random', [1, 3, 224, 224]);
  }

  if (inputs === 'img224-f16') {
    getFeedInfo(inputNames[0], 'float16', 'random', [1, 3, 224, 224]);
  }

  if (inputs === 'img224-i8') {
    getFeedInfo(inputNames[0], 'int8', 'random', [1, 3, 224, 224]);
  }

  if (inputs === 'llm-decoder') {
    if (modelName === 'gpt2-decoder') {
      seqLen = 8;
    } else if (['distilgpt2-decoder', 'distilgpt2-decoder-merged'].indexOf(modelName) >= 0) {
      seqLen = 16;
    }

    getFeedInfo('input_ids', 'int64', 99n, [batchSize, seqLen]);
    getFeedInfo('attention_mask', 'int64', 1n, [batchSize, seqLen]);
    getPastKeyValuesInfo([batchSize, 12, seqLen, 64]);
  }

  if (inputs === 'm2m100-decoder') {
    getFeedInfo('encoder_attention_mask', 'int64', 1n, [1, encSeqLen]);
    getFeedInfo('input_ids', 'int64', 99n, [1, decSeqLen]);
    getFeedInfo('encoder_hidden_states', 'float32', 1, [1, encSeqLen, 1024]);
    getPastKeyValuesInfo([1, 16, seqLen, 64]);
  }

  if (inputs === 'm2m100-encoder') {
    getFeedInfo('input_ids', 'int64', 99n, [1, encSeqLen]);
    getFeedInfo('attention_mask', 'int64', 1n, [1, encSeqLen]);
  }

  if (inputs === 'mobilenetv3') {
    getFeedInfo(inputNames[0], 'float32', 'random', [1, 224, 224, 3]);
  }

  if (inputs === 'realesrgan') {
    const modelInfo = modelName.split('-');
    const tileSize = parseInt(modelInfo[1].replace('t', ''));
    const dataType = modelName.endsWith('f16') ? '16' : '32';
    getFeedInfo(`in_image_float${dataType}_rgb01`, `float${dataType}`, 'random', [1, 3, tileSize, tileSize]);
  }

  if (inputs === 'sam-b-vision-encoder') {
    getFeedInfo('pixel_values', 'float32', 'random', [1, 3, 1024, 1024]);
  }

  if (inputs === 'sam-decoder') {
    getFeedInfo('image_embeddings', 'float32', 'random', [1, 256, 64, 64]);
    getFeedInfo('point_coords', 'float32', 'random', [1, 2, 2]);
    getFeedInfo('point_labels', 'float32', 'random', [1, 2]);
    getFeedInfo('mask_input', 'float32', 'random', [1, 1, 256, 256]);
    getFeedInfo('has_mask_input', 'float32', 'random', [1]);
    getFeedInfo('orig_im_size', 'float32', [512, 512], [2]);
  }

  if (inputs === 'sam-decoder-f16') {
    getFeedInfo('image_embeddings', 'float16', 'random', [1, 256, 64, 64]);
    getFeedInfo('point_coords', 'float16', 'random', [1, 2, 2]);
    getFeedInfo('point_labels', 'float16', 'random', [1, 2]);
    getFeedInfo('mask_input', 'float16', 'random', [1, 1, 256, 256]);
    getFeedInfo('has_mask_input', 'float16', 'random', [1]);
    getFeedInfo('orig_im_size', 'float32', [512, 512], [2]);
  }

  if (inputs === 'sam-encoder') {
    getFeedInfo('input_image', 'float32', 1, [224, 224, 3]);
  }

  if (inputs === 'sd-text-encoder') {
    getFeedInfo('input_ids', 'int32', 99, [1, encSeqLen]);
  }

  if (inputs === 'sd-unet') {
    getFeedInfo('sample', 'float32', 1, [1, 4, 64, 64]);
    getFeedInfo('timestep', 'int64', 1n, [1]);
    getFeedInfo('encoder_hidden_states', 'float32', 1, [1, 77, 768]);
  }

  if (inputs === 'sd-unet-f16') {
    getFeedInfo('sample', 'float16', 1, [1, 4, 64, 64]);
    getFeedInfo('timestep', 'int64', 1n, [1]);
    getFeedInfo('encoder_hidden_states', 'float16', 1, [1, 77, 768]);
  }

  if (inputs === 'sd-vae-decoder-f16') {
    getFeedInfo('latent_sample', 'float16', 'random', [1, 4, 64, 64]);
  }

  if (inputs === 'sd-vae-decoder') {
    getFeedInfo('latent_sample', 'float32', 'random', [1, 4, 64, 64]);
  }

  if (inputs === 'sd-vae-encoder') {
    getFeedInfo('sample', 'float32', 'random', [1, 3, 512, 512]);
  }

  if (inputs === 't5-decoder' || inputs === 'mt5-decoder' || inputs === 'flan-t5-decoder') {
    getFeedInfo('encoder_attention_mask', 'int64', 1n, [batchSize, encSeqLen]);
    getFeedInfo('encoder_hidden_states', 'float32', 'random', [batchSize, encSeqLen, 512]);
    getFeedInfo('input_ids', 'int64', 99n, [batchSize, decSeqLen]);
    const dims = inputs === 't5-decoder' ? [batchSize, 8, decSeqLen, 64] : [batchSize, 6, decSeqLen, 64];
    getPastKeyValuesInfo(dims);
  }

  if (inputs === 't5-encoder') {
    getFeedInfo('input_ids', 'int64', 99n, [1, decSeqLen]);
  }

  if (inputs === 'vit-gpt2-image-captioning-decoder') {
    decSeqLen = 168;
    getFeedInfo('input_ids', 'int64', 99n, [1, 168]);
    getFeedInfo('encoder_hidden_states', 'float32', 'random', [1, 168, 768]);
    getPastKeyValuesInfo([1, 12, decSeqLen, 64]);
  }

  if (inputs === 'whisper-decoder') {
    getFeedInfo('input_ids', 'int64', 1n, [1, 1]);
    getFeedInfo('encoder_hidden_states', 'float32', 'random', [1, 1500, 384]);
    getPastKeyValuesInfo([1, 6, 1500, 64]);
  }

  getFeedInfo('use_cache_branch', 'bool', true);

  if (isDict(inputs)) {
    for (let key in inputs) {
      let value = inputs[key];
      getFeedInfo(key, value[0], value[1], value[2]);
    }
  }
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
      'encoder_sequence_length / 2': 1500,
    };
  }

  return freeDimensionOverrides;
}

function getGraphCaptureInfo(modelName) {
  if ([
        'densenet-9',
        'detr-resnet-50',
        'dino-vitb16',
        'efficientnet-lite4-11',
        'emotion-ferplus-8',
        'mobilenetv2-12',
        'mobilenetv2-12-f16',
        'mobilenetv3-small-100',
        'mobilevit-small',
        'resnet50-v2-7',
        'sam-b-encoder',
        'tinyyolov2-8',
        'vit-base-patch16-224',
        'vit-gpt2-image-captioning-encoder',
        'whisper-tiny-encoder',
      ].indexOf(modelName) >= 0) {
    return true;
  } else {
    return false;
  }
}

function getModelFolderInfo(modelName) {
  modelFolder = '';
  if (['sd-unet-f16', 'sd-vae-decoder-arthur', 'sd-vae-decoder-f16'].indexOf(modelName) >= 0) {
    modelFolder = 'private/';
  } else if (['sam-b-vision-encoder'].indexOf(modelName) >= 0) {
    modelFolder = 'tmp/';
  }
  return modelFolder;
}

function isDict(v) {
  return typeof v === 'object' && v !== null && !(v instanceof Array) && !(v instanceof Date);
}
