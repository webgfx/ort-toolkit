# ort-toolkit

Visit https://webatintel.github.io/ort-toolkit for more details

## Run WebGPU

- Start Chrome with "--enable-dawn-features=allow_unsafe_apis,use_dxc --enable-features=SharedArrayBuffer"
- Example: https://webatintel.github.io/ort-toolkit/?tasks=performance&ep=webgpu&modelName=mobilenetv2-12&modelUrl=hf&enableReadback=true

## Run WebNN

- Start Chrome with "--enable-features=MachineLearningNeuralNetworkService --enable-experimental-web-platform-features"
- Example: https://webatintel.github.io/ort-toolkit/?tasks=performance&ep=webnn-gpu&modelName=mobilenetv2-12&modelUrl=hf&enableReadback=true
