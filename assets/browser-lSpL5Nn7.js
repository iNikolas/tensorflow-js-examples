import{K as T,N as v,O as P,c as k,T as D,b6 as x,d6 as I,E as y,d7 as C,o as M,d8 as b}from"./register_all_kernels-BMJVJEs3.js";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function H(e,t,n){if(T(e),t!=null&&t.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const a=v(e,n);if(a.length!==3&&a.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(a.length===1&&t==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return P(e,t,a,n)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let c,E=!1;function L(e,t=3){if(t>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(e==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,a=!1,l=!1,h=!1,m=!1,d=!1;if(e.data instanceof Uint8Array)n=!0;else if(typeof ImageData<"u"&&e instanceof ImageData)a=!0;else if(typeof HTMLVideoElement<"u"&&e instanceof HTMLVideoElement)l=!0;else if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement)h=!0;else if(e.getContext!=null)m=!0;else if(typeof ImageBitmap<"u"&&e instanceof ImageBitmap)d=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);if(I(b,y.backendName)!=null){const g={pixels:e},w={numChannels:t};return y.runKernel(b,g,w)}const[o,r]=l?[e.videoWidth,e.videoHeight]:[e.width,e.height];let s;if(m)s=e.getContext("2d").getImageData(0,0,o,r).data;else if(a||n)s=e.data;else if(h||l||d){if(c==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")c=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else c=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});c.canvas.width=o,c.canvas.height=r,c.drawImage(e,0,0,o,r),s=c.getImageData(0,0,o,r).data}let f;if(t===4)f=new Int32Array(s);else{const g=o*r;f=new Int32Array(g*t);for(let w=0;w<g;w++)for(let p=0;p<t;++p)f[w*t+p]=s[w*4+p]}return H(f,[r,o,t],"int32")}function N(e){if(e.rank!==2&&e.rank!==3)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${e.rank}.`);const t=e.rank===2?1:e.shape[2];if(t>4||t===2)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${t}`);if(e.dtype!=="float32"&&e.dtype!=="int32")throw new Error(`Unsupported type for toPixels: ${e.dtype}. Please use float32 or int32 tensors.`)}async function A(e,t){let n=k(e,"img","toPixels");if(!(e instanceof D)){const o=n;n=x(o,"int32"),o.dispose()}N(n);const[a,l]=n.shape.slice(0,2),h=n.rank===2?1:n.shape[2],m=await n.data(),d=n.dtype==="float32"?255:1,u=new Uint8ClampedArray(l*a*4);for(let o=0;o<a*l;++o){const r=[0,0,0,255];for(let f=0;f<h;f++){const i=m[o*h+f];if(n.dtype==="float32"){if(i<0||i>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${i}.`)}else if(n.dtype==="int32"&&(i<0||i>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${i}.`);h===1?(r[0]=i*d,r[1]=i*d,r[2]=i*d):r[f]=i*d}const s=o*4;u[s+0]=Math.round(r[0]),u[s+1]=Math.round(r[1]),u[s+2]=Math.round(r[2]),u[s+3]=Math.round(r[3])}if(t!=null){E||I(C,y.backendName)!=null&&(console.warn("tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead."),E=!0),t.width=l,t.height=a;const o=t.getContext("2d"),r=new ImageData(u,l,a);o.putImageData(r,0,0)}return n!==e&&n.dispose(),u}const O=M({fromPixels_:L});export{H as a,O as f,A as t};
