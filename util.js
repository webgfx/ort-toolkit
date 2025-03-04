const displayPrecision = 2;


// Get model via Origin Private File System
async function getOPFS(name, url, updateModel) {
  async function updateFile(needCache) {
    const response = await fetch(url);
    const buffer = await readResponse(response);
    if (!needCache) {
      return buffer;
    }
    fileHandle = await root.getFileHandle(name, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(buffer);
    await writable.close();
    return buffer;
  }

  if (!navigator.storage) {
    return await updateFile(needCache = false);
  }

  const root = await navigator.storage.getDirectory();
  let fileHandle;

  if (updateModel) {
    return await updateFile(needCache = true);
  }

  try {
    fileHandle = await root.getFileHandle(name);
    const blob = await fileHandle.getFile();
    return await blob.arrayBuffer();
  } catch (e) {
    return await updateFile(needCache = true);
  }
}

// Get model via Cache API
async function getCache(name, url, updateModel) {
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
  return data.reduce((accumulator, currentValue) => { return accumulator + currentValue }, 0);
}

function toggleClass(el, className) {
  if (el.className.indexOf(className) >= 0) {
    el.className = el.className.replace(className, '');
  } else {
    el.className += className;
  }
}

function compare(actual, expected, epsilons) {
  try {
    areCloseObjects(actual, expected, epsilons);
  } catch (e) {
    console.error(e);
    return false;
  }
  return true;
}

function areCloseObjects(actual, expected, epsilons) {
  let actualKeys = Object.keys(actual);
  let expectedKeys = Object.keys(expected);
  if (actualKeys.length != expectedKeys.length) {
    throw new Error(`Actual length ${actualKeys.length} not equal Expected length ${expectedKeys.length}`);
  }
  for (let i = 0; i < actualKeys.length; i++) {
    let key = actualKeys[i];
    let isArray = isTypedArray(actual[key]) && isTypedArray(expected[key]);
    let isObject = typeof (actual[key]) === 'object' && typeof (expected[key]) === 'object';
    if (isArray) {
      areCloseArrays(actual[key], expected[key], epsilons);
    } else if (isObject) {
      areCloseObjects(actual[key], expected[key], epsilons);
    } else {
      if (!areClosePrimitives(actual[key], expected[key], epsilons)) {
        throw new Error(`Objects differ: actual[${key}] = ${JSON.stringify(actual[key])}, expected[${key}] = ${JSON.stringify(expected[key])}!`);
      }
    }
  }
  return true;
}

function areCloseArrays(actual, expected, epsilons) {
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

    if (!areClosePrimitives(a, e, epsilons)) {
      throw new Error(
        `Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
        `Actual:   ${actualFlat}.\n` +
        `Expected: ${expectedFlat}.`);
    }
  }
}

function areClosePrimitives(actual, expected, epsilons) {
  if (isNaN(actual) || isNaN(expected)) {
    return false;
  } else if (!isFinite(actual) && !isFinite(expected)) {
    return true;
  }

  const error = Math.abs(actual - expected);
  if (Math.abs(actual) >= 1) {
    errorRatio = error / Math.min(Math.abs(actual), Math.abs(expected));
    if ((error > epsilons[0]) || errorRatio > epsilons[1]) {
      console.error(`actual=${actual}, expected=${expected}`);
      return false;
    }
  } else {
    if (error > epsilons[1]) {
      console.error(`actual=${actual}, expected=${expected}`);
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
  int64: BigInt64Array,
  bool: Uint8Array,
};

function clone(x) {
  let feed = {};
  for (const [key, value] of Object.entries(x)) {
    let func = type_to_func[value.type];
    let arrayType = func.from(value.data);
    feed[key] = new ort.Tensor(value.type, arrayType.slice(0), value.dims);
  }
  return feed;
}

// https://gist.github.com/mfirmin/456e1c6dcf7b0e1bda6e940add32adad
// This function converts a Float16 stored as the bits of a Uint16 into a Javascript Number.
function float16ToNumber(input) {
  // Create a 32 bit DataView to store the input
  const arr = new ArrayBuffer(4);
  const dv = new DataView(arr);

  // Set the Float16 into the last 16 bits of the dataview
  // So our dataView is [00xx]
  dv.setUint16(2, input, false);

  // Get all 32 bits as a 32 bit integer
  // (JS bitwise operations are performed on 32 bit signed integers)
  const asInt32 = dv.getInt32(0, false);

  // All bits aside from the sign
  let rest = asInt32 & 0x7fff;
  // Sign bit
  let sign = asInt32 & 0x8000;
  // Exponent bits
  const exponent = asInt32 & 0x7c00;

  // Shift the non-sign bits into place for a 32 bit Float
  rest <<= 13;
  // Shift the sign bit into place for a 32 bit Float
  sign <<= 16;

  // Adjust bias
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding
  rest += 0x38000000;
  // Denormals-as-zero
  rest = (exponent === 0 ? 0 : rest);
  // Re-insert sign bit
  rest |= sign;

  // Set the adjusted float32 (stored as int32) back into the dataview
  dv.setInt32(0, rest, false);

  // Get it back out as a float32 (which js will convert to a Number)
  const asFloat32 = dv.getFloat32(0, false);

  return asFloat32;
}

function renderData(heads, data, title) {
  let row, th, td;

  // title
  let h = document.createElement("h3");
  h.innerHTML = title;
  h.align = "center";
  h.className = "replacable";
  document.body.appendChild(h);

  // table
  let table = document.createElement("table");
  table.className = "sortable replacable";
  table.align = "center";
  table.style.width = "80%";
  table.setAttribute("border", "1");
  document.body.appendChild(table);

  // thead
  let header = table.createTHead("thead");
  row = header.insertRow(0);
  row.style.fontWeight = "bold";
  for (let head of heads) {
    let th = document.createElement("th");
    th.innerHTML = head;
    row.appendChild(th);
  }

  // tbody
  let tbody = document.createElement("tbody");
  table.appendChild(tbody);
  // rest of line
  for (let i = 0; i < data.length; ++i) {
    let rowInfo = data[i];
    row = tbody.insertRow(i);
    row.onclick = function () {
      toggleClass(this, "highlight");
    };
    for (let j = 0; j < heads.length; j++) {
      td = row.insertCell(j);
      let cellInfo = rowInfo[j];
      if (heads[j].startsWith("Time")) {
        cellInfo = cellInfo.toFixed(displayPrecision);
      }
      td.innerHTML = cellInfo;
    }
  }

  // tfoot
  let needTfoot = false;
  for (let i = 0; i < heads.length; ++i) {
    if (heads[i].startsWith("Time")) {
      needTfoot = true;
      break;
    }
  }
  if (needTfoot) {
    let tfoot = document.createElement("tfoot");
    table.appendChild(tfoot);
    row = tfoot.insertRow(0);
    row.style.fontWeight = "bold";
    let sums = new Array(heads.length).fill("");
    sums[0] = "Sum";
    for (let i = 0; i < heads.length; ++i) {
      if (!heads[i].startsWith("Time")) {
        continue;
      }

      let sum = 0;
      for (let j = 0; j < data.length; j++) {
        sum += data[j][i];
      }
      sums[i] = sum.toFixed(displayPrecision);
    }
    for (let i = 0; i < heads.length; ++i) {
      td = row.insertCell(i);
      td.innerHTML = sums[i];
    }
  }

  // blank line
  document.body.appendChild(document.createElement("p"));
}

function renderAggregatedData(heads, data, title) {
  let kernelTime = {};
  for (let d of data) {
    let kernel = d[1];
    if (!(kernel in kernelTime)) {
      kernelTime[kernel] = d[2];
    } else {
      kernelTime[kernel] += d[2];
    }
  }
  let totalTime = getSum(Object.values(kernelTime));
  let keys = Object.keys(kernelTime);
  let sortedKernelTime = keys.sort(function (a, b) {
    return kernelTime[b] - kernelTime[a];
  });
  let sortedAggregatedData = [];
  for (let kernel of sortedKernelTime) {
    let time = kernelTime[kernel];
    sortedAggregatedData.push([kernel, time, ((time / totalTime) * 100).toFixed(2)]);
  }

  renderData(heads, sortedAggregatedData, title);
}

function getParam(name, type, _default) {
  name = name.replace(/[\[\]]/g, "\\$&");
  let regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)", "i");
  let results = regex.exec(window.location.href);
  if (!results || !results[2]) return _default;

  const result = decodeURIComponent(results[2].replace(/\+/g, " "));
  if (type === "Boolean") {
    if (result === "true") {
      return true;
    } else if (result === "false") {
      return false;
    }
  } else if (type === "Number") {
    return parseInt(result);
  } else {
    return result;
  }
}

async function loadScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.onload = resolve;
    script.onerror = reject;
    script.src = url;
    if (url.startsWith("http")) {
      script.crossOrigin = "anonymous";
    }
    document.body.append(script);
  });
}
