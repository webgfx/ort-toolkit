
// Get model via Origin Private File System
async function getModelOPFS(name, url, updateModel) {
  const root = await navigator.storage.getDirectory();
  let fileHandle;

  async function updateFile() {
    const response = await fetch(url);
    const buffer = await readResponse(response);
    fileHandle = await root.getFileHandle(name, {create: true});
    const writable = await fileHandle.createWritable();
    await writable.write(buffer);
    await writable.close();
    return buffer;
  }

  if (updateModel) {
    return await updateFile();
  }

  try {
    fileHandle = await root.getFileHandle(name);
    const blob = await fileHandle.getFile();
    return await blob.arrayBuffer();
  } catch(e) {
    return await updateFile();
  }
}

// Get model via Cache API
async function getModelCache(name, url, updateModel) {
  const cache = await caches.open(name);
  if (updateModel) {
    await cache.add(url);
  }
  let response = await cache.match(url);
  if (!response) {
    await cache.add(url);
    response = await cache.match(url);
  }
  const buffer = await readResponse(response);
  return buffer;
}

async function readResponse(response) {
  const contentLength = response.headers.get('Content-Length');
  let total = parseInt(contentLength ?? '0');
  let buffer = new Uint8Array(total);
  let loaded = 0;

  const reader = response.body.getReader();
  async function read() {
    const { done, value } = await reader.read();
    if (done) return;

    let newLoaded = loaded + value.length;
    if (newLoaded > total) {
      total = newLoaded;
      let newBuffer = new Uint8Array(total);
      newBuffer.set(buffer);
      buffer = newBuffer;
    }
    buffer.set(value, loaded);
    loaded = newLoaded;
    return read();
  }

  await read();
  return buffer;
}

function reportStatus(status) {
  document.getElementById('status').innerHTML = status;
}

function getSum(data) {
  return data.reduce((accumulator, currentValue) => {
    return accumulator + currentValue
  }, 0);
}

function toggleClass(el, className) {
  if (el.className.indexOf(className) >= 0) {
    el.className = el.className.replace(className, '');
  } else {
    el.className += className;
  }
}

function compare(actual, expected, epsilon = 1e-3) {
  try {
    areCloseObjects(actual, expected, epsilon);
  } catch (e) {
    return false;
  }
  return true;
}

function areCloseObjects(actual, expected, epsilon) {
  let actualKeys = Object.getOwnPropertyNames(actual);
  let expectedKeys = Object.getOwnPropertyNames(expected);
  if (actualKeys.length != expectedKeys.length) {
    throw new Error(`Actual length ${actualKeys.length} not equal Expected length ${expectedKeys.length}`);
  }
  for (let i = 0; i < actualKeys.length; i++) {
    let key = actualKeys[i];
    let isArray = isTypedArray(actual[key]) && isTypedArray(expected[key]);
    let isObject = typeof (actual[key]) === 'object' && typeof (expected[key]) === 'object';
    if (isArray) {
      areCloseArrays(actual[key], expected[key], epsilon);
    } else if (isObject) {
      areCloseObjects(actual[key], expected[key], epsilon);
    } else {
      if (!areClosePrimitives(actual[key], expected[key])) {
        throw new Error(`Objects differ: actual[${key}] = ${JSON.stringify(actual[key])}, expected[${key}] = ${JSON.stringify(expected[key])}!`);
      }
    }
  }
  return true;
}

function areCloseArrays(actual, expected, epsilon) {
  let checkClassType = true;
  if (isTypedArray(actual) || isTypedArray(expected)) {
    checkClassType = false;
  }
  if (isTypedArray(actual) && isTypedArray(expected)) {
    checkClassType = true;
  }
  if (checkClassType) {
    const aType = actual.constructor.name;
    const bType = expected.constructor.name;

    if (aType !== bType) {
      throw new Error(`Arrays are of different type. Actual: ${aType}. Expected: ${bType}`);
    }
  }

  const actualFlat = isTypedArray(actual) ? actual : flatten(actual);
  const expectedFlat = isTypedArray(expected) ? expected : flatten(expected);

  if (actualFlat.length !== expectedFlat.length) {
    throw new Error(
      `Arrays have different lengths actual: ${actualFlat.length} vs ` +
      `expected: ${expectedFlat.length}.\n` +
      `Actual:   ${actualFlat}.\n` +
      `Expected: ${expectedFlat}.`);
  }
  for (let i = 0; i < expectedFlat.length; ++i) {
    const a = actualFlat[i];
    const e = expectedFlat[i];

    if (!areClosePrimitives(a, e)) {
      throw new Error(
        `Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
        `Actual:   ${actualFlat}.\n` +
        `Expected: ${expectedFlat}.`);
    }
  }
}

function areClosePrimitives(actual, expected, epsilon) {
  if (!isFinite(actual) && !isFinite(expected)) {
    return true;
  } else if (isNaN(actual) || isNaN(expected)) {
    return false;
  }

  const error = Math.abs(actual - expected);
  if (Math.abs(actual) >= 1) {
    if ((error > 1e-1) || error / Math.min(Math.abs(actual), Math.abs(expected)) > epsilon) {
      return false;
    }
  } else {
    if (error > epsilon) {
      return false;
    }
  }
  return true;
}

function isTypedArray(object) {
  return ArrayBuffer.isView(object) && !(object instanceof DataView);
}

const type_to_func = {
  float32: Float32Array,
  uint16: Uint16Array,
  float16: Uint16Array,
  int32: Int32Array,
  BigInt64Array: BigInt64Array,
};

function clone(x) {
  let feed = {};
  for (const [key, value] of Object.entries(x)) {
    let func = type_to_func[value.type];
    let arrayType = func.from(value.data);
    feed[key] = new ort.Tensor(
      value.type,
      arrayType.slice(0),
      value.dims
    );
  }
  return feed;
}