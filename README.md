# ort-toolkit

This is a toolkit to run ONNX Runtime tasks, including artifact, conformance, performance, webgpuProfiling, ortProfiling and so on.

- artifact: Get the uniform info
- conformance: Compare the result with wasm backend
- performance: Performance test
- webgpuProfiling: Get execution time of each op
- ortProfiling: Understand the Execution Provider for each op

## Start Chrome Browser

If you encounter cross origin issue, you may install Chrome extension "CORS Unblock" (https://chrome.google.com/webstore/detail/cors-unblock/lfhmikememgdcahcdlaciloancbhjino?hl=en) and enable it.

To manage OPFS (Origin Private File System), you may install Chrome extension OPFS Explorer: https://chrome.google.com/webstore/detail/opfs-explorer/acndjpgkpaclldomagafnognkcgjignd

[WebGPU]

Start Chrome with "--enable-dawn-features=allow_unsafe_apis,use_dxc --enable-features=SharedArrayBuffer"

- --enable-dawn-features=allow_unsafe_apis: Make timestamp query work
- --enable-dawn-features=use_dxc: Enable DXC instead of FXC for WGSL compilation
- --enable-features=SharedArrayBuffer: Enable SharedArrayBuffer otherwise you may get 'TypeError: Cannot convert a BigInt value to a number'

Example: https://webatintel.github.io/ort-toolkit/?tasks=performance&ep=webgpu&modelName=mobilenetv2-12&modelUrl=hf&enableReadback=true

[WebNN]

Start Chrome with "--enable-features=MachineLearningNeuralNetworkService --enable-experimental-web-platform-features --disable-gpu-sandbox"

Example: https://webatintel.github.io/ort-toolkit/?tasks=performance&ep=webnn&device=gpu&modelName=mobilenetv2-12&modelUrl=hf&enableReadback=true

## Usage

If your web server supports wasm multiple threads, ort-wasm-simd-threaded.jsep.[js|wasm] will be called by ort.webgpu.min.js, which may be different from your build. You may set wasmThreads=1 to fall back to ort-wasm-simd.jsep.[js|wasm].

Some parameters are supported in url, and you may use them as 'index.html?key0=value0&key1=value1...'. Supported parameters are:

- deviceType: device type, which can be gpu or cpu
- ep: execution provider. E.g., webgpu, wasm
- layout=[NCHW|NHWC]. NHWC is the default.
- modelName: name of modelName. E.g., mobilenetv2-12
- modelUrl=[hf|server|wp-27|[url]]. Note that if you provide the url, you may fail to execute the modelName as the
  inputs are not defined well.

- ortUrl: ort url. Example: ortUrl=https://wp-27.sh.intel.com/workspace/project/onnxruntime or ortUrl=gh/20231129
- runTimes: Run times
- tasks=[task0,task1,task2]: tasks to run, split by ','. Candidates are 'conformance', 'performance',
  'ortProfiling', 'webgpuProfiling'.

- updateModel=[true|false]. False (default) means no update.
- warmupTimes: Warmup times
- wasmThreads: wasm threads number
- webnnNumThreads: WebNN numThreads for cpu

## Examples

- Conformance

  https://wp-27.sh.intel.com/workspace/project/onnxruntime/ort-toolkit/?tasks=conformance&modelName=mobilenetv2-12&ep=webgpu&ortUrl=https://wp-27.sh.intel.com/workspace/project/onnxruntime&warmupTimes=10&runTimes=10
