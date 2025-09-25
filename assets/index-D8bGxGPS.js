import{r as q,j as g,R as xe}from"./index-C17uvJyM.js";import{c as _e}from"./helpers-Cs5YJYh0.js";import{u as Cs}from"./hooks-C0HuhxgG.js";import{M as Ds}from"./view-BhGDTypU.js";import{o as O,b as v,c as w,d as le,E as C,A as Ps,e as De,m as Z,f as re,g as M,h as ue,j as ve,k as ze,B as js,l as xs,D as $s,n as Fs,L as zs,p as E,q as ce,S as Rs,M as Ls,T as pe,u as oe,v as wt,w as Vs,x as fe,R as Bs,y as qs,z as Us,C as Pe,F as Hs,G as Qt,H as Kt,r as Jt,I as ye,J as Tt,K as rt,N as nt,O as it,P as Xt,Q as Ws,U as Gs,V as Yt,W as Zt,X as Mt,Y as Qs,Z as Q,_ as Oe,$ as es,a0 as ts,a1 as Ks,a2 as Js,a3 as Xs,a4 as Ys,a5 as ne,a6 as Zs,a7 as ss,a8 as Ms,a9 as ea,aa as ta,ab as sa,ac as aa,ad as ra,ae as St,af as vt,ag as na,ah as ia,ai as oa,aj as ua,ak as la,al as ca,am as pa,an as ot,ao as Ot,ap as ut,aq as X,ar as je,as,at as lt,au as ma,av as rs,aw as Re,ax as ha,ay as da,az as ns,aA as fa,aB as ya,aC as ga,aD as ba,aE as Na,aF as wa,aG as Ta,aH as Sa,aI as va,aJ as Oa,aK as is,aL as L,aM as ka,aN as _a,aO as Ea,aP as Aa,aQ as Ia,aR as Ca,aS as Da,aT as Pa,aU as ja,aV as xa,aW as $a,aX as Fa,aY as za,aZ as Ra,a_ as La,a$ as Va,b0 as Ba,b1 as qa,b2 as Ua,b3 as Ha,b4 as Wa,b5 as Ga,b6 as Qa,b7 as Ka,b8 as Ja,b9 as Xa,ba as Ya,bb as Za,bc as Ma,bd as er,be as tr,bf as sr,bg as ar,bh as rr,bi as nr,bj as ir,bk as or,bl as ur,bm as lr,bn as cr,bo as pr,bp as mr,bq as hr,br as dr,bs as fr,bt as yr,bu as gr,bv as br,bw as Nr,bx as wr,by as Tr,bz as Sr,bA as vr,bB as Or,bC as kr,bD as _r,bE as Er,bF as Ar,bG as Ir,bH as Cr,bI as Dr,bJ as Pr,bK as jr,bL as xr,i as os,bM as $r,bN as Fr,bO as zr,bP as Rr,bQ as Lr,bR as Vr,bS as Br,bT as qr,bU as Ur,bV as Hr,bW as Wr,bX as Gr,bY as Qr,bZ as Kr,b_ as Jr,b$ as Xr,c0 as Yr,c1 as Zr,c2 as Mr,c3 as en,c4 as tn,c5 as sn,c6 as an,c7 as rn,c8 as nn,c9 as on,ca as un,cb as ln,cc as cn,cd as pn,ce as mn,cf as hn,cg as dn,ch as fn,ci as yn,cj as gn,ck as bn,cl as Nn,cm as wn,cn as Tn,co as Sn,cp as vn,cq as On,cr as kn,cs as _n,ct as En,cu as An,cv as In,cw as Cn,cx as Dn,cy as Pn,cz as jn,cA as xn,cB as $n,cC as Fn,cD as zn,cE as Rn,cF as Ln,cG as Vn,cH as Bn,cI as qn,s as Un,cJ as Hn,cK as Wn,cL as Gn,a as me,cM as Qn,cN as Kn,cO as Jn,cP as Xn,cQ as Yn,cR as Zn,cS as Mn,cT as ei,cU as us,cV as ti,cW as si,cX as ai,cY as ri,cZ as ge,c_ as ni,c$ as ii,d0 as oi,d1 as ui,d2 as G,t as U,d3 as Ee,d4 as li,d5 as ci}from"./register_all_kernels-DXH-6K2q.js";import{a as pi,f as mi}from"./browser-qhGOZmkM.js";/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function hi(t){v(Array.isArray(t),()=>"The argument passed to tf.addN() must be a list of tensors"),v(t.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${t.length}`);const e=t.map((r,n)=>w(r,`tensors${n}`,"addN")),s=e[0];e.forEach(r=>{if(r.dtype!==s.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),e.forEach(r=>{if(!le(r.shape,s.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const a=e;return C.runKernel(Ps,a)}const di=O({addN_:hi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function fi(t,e,s,a,r,n){const u=w(t,"forgetBias","basicLSTMCell"),o=w(e,"lstmKernel","basicLSTMCell"),l=w(s,"lstmBias","basicLSTMCell"),c=w(a,"data","basicLSTMCell"),p=w(r,"c","basicLSTMCell"),m=w(n,"h","basicLSTMCell"),h=De([c,m],1),d=Z(h,o),b=re(d,l),y=b.shape[0],f=b.shape[1]/4,N=[y,f],S=M(b,[0,0],N),k=M(b,[0,f],N),T=M(b,[0,f*2],N),_=M(b,[0,f*3],N),j=re(ue(ve(S),ze(k)),ue(p,ve(re(u,T)))),A=ue(ze(j),ve(_));return[j,A]}const yi=O({basicLSTMCell_:fi});/**
 * @license
 * Copyright 2023 Google LLC.
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
 */function gi(t,e){const s=w(t,"x","bitwiseAnd"),a=w(e,"y","bitwiseAnd");if(!le(s.shape,a.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${s.shape}, y: ${a.shape}`);if(s.dtype!=="int32"||a.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${s.dtype} and type of y: ${a.dtype}`);const r={a:s,b:a};return C.runKernel(js,r)}const bi=O({bitwiseAnd_:gi});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
 */function Ni(t,e){const s=w(t,"s0","broadcastArgs","int32"),a=w(e,"s1","broadcastArgs","int32");if(s.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${s.rank}`);if(a.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${a.rank}`);const r={s0:s,s1:a};return C.runKernel(xs,r)}const wi=O({broadcastArgs_:Ni});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function Ti(t){const s={x:w(t,"x","diag")};return C.runKernel($s,s)}const Si=O({diag_:Ti});/**
 * @license
 * Copyright 2023 Google LLC.
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
 */function vi(t,e){const s=w(t,"x","ensureShape","string_or_numeric");if(!Fs(s.shape,e))throw new Error(`EnsureShape: Shape of tensor ${s.shape} is not compatible with expected shape ${e}`);return t}const Oi=O({ensureShape_:vi});/**
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
 */function ki(t,e,s){if(s<=0)throw new Error("The number of values should be positive.");const a={start:t,stop:e,num:s};return C.runKernel(zs,{},a)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
 */const Ne=2147483648;function _i(t,e,s="left"){const a=w(t,"sortedSequence","searchSorted"),r=w(e,"values","searchSorted"),n=a.shape[a.shape.length-1],u=r.shape[r.shape.length-1],o=E(a,[-1,n]),l=E(r,[-1,u]);if(o.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(o.shape[0]!==l.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(ce(l.shape)>=Ne)throw new Error(`values tensor size must less than ${Ne}`);if(o.shape[1]>=Ne)throw new Error(`trailing dim_size must less than ${Ne} for int32 output type, was ${o.shape[1]}`);const c={sortedSequence:o,values:l},p={side:s};return C.runKernel(Rs,c,p)}const ct=O({searchSorted_:_i});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
 */function Ei(t,e){return ct(t,e,"left")}/**
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
 */function Ai(t,e,s,a,r=!1){const u={x:w(t,"x","maxPoolWithArgmax")},o={filterSize:e,strides:s,pad:a,includeBatchInIndex:r},l=C.runKernel(Ls,u,o);return{result:l[0],indexes:l[1]}}const Ii=O({maxPoolWithArgmax_:Ai});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
 */function Ci(t,e,{indexing:s="xy"}={}){if(s!=="xy"&&s!=="ij")throw new TypeError(`${s} is not a valid third argument to meshgrid`);if(t===void 0)return[];let a=w(t,"x","meshgrid",t instanceof pe?t.dtype:"float32");if(e===void 0)return[a];let r=w(e,"y","meshgrid",e instanceof pe?e.dtype:"float32");const n=ce(a.shape),u=ce(r.shape);return s==="xy"?(a=E(a,[1,-1]),r=E(r,[-1,1]),[Z(oe([u,1],a.dtype),a),Z(r,oe([1,n],r.dtype))]):(a=E(a,[-1,1]),r=E(r,[1,-1]),[Z(a,oe([1,u],a.dtype)),Z(oe([n,1],r.dtype),r)])}function Di(t,e,s,a){const r=w(e,"data","multiRNNCell"),n=wt(s,"c","multiRNNCell"),u=wt(a,"h","multiRNNCell");let o=r;const l=[];for(let m=0;m<t.length;m++){const h=t[m](o,n[m],u[m]);l.push(h[0]),l.push(h[1]),o=h[1]}const c=[],p=[];for(let m=0;m<l.length;m+=2)c.push(l[m]),p.push(l[m+1]);return[c,p]}const Pi=O({multiRNNCell_:Di});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function ji(t,e,s,a=!1){const r=w(t,"logits","multinomial"),n=r.size,u=r.rank;if(n<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${n}.`);if(u>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${u}`);s=s||Math.random();const l={logits:u===1?E(r,[1,-1]):r},c={numSamples:e,seed:s,normalized:a},p=C.runKernel(Vs,l,c);return u===1?E(p,[p.size]):p}const xi=O({multinomial_:ji});function $i(t,e){const s=w(t,"v1","outerProduct"),a=w(e,"v2","outerProduct");v(s.rank===1&&a.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${s.rank} and ${a.rank}.`);const r=E(s,[-1,1]),n=E(a,[1,-1]);return Z(r,n)}const Fi=O({outerProduct_:$i});function zi(t,e,s=0){return v(e.length===2,()=>"Invalid number of paddings. Must be length of 2."),fe(t,[e],s)}const Ri=O({pad1d_:zi});function Li(t,e,s=0){return v(e.length===2&&e[0].length===2&&e[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),fe(t,e,s)}const Vi=O({pad2d_:Li});function Bi(t,e,s=0){return v(e.length===3&&e[0].length===2&&e[1].length===2&&e[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),fe(t,e,s)}const qi=O({pad3d_:Bi});function Ui(t,e,s=0){return v(e.length===4&&e[0].length===2&&e[1].length===2&&e[2].length===2&&e[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),fe(t,e,s)}const Hi=O({pad4d_:Ui});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
 */function Wi(t,e,s,a){const r=t.map((p,m)=>w(p,`tensors${m}`,"raggedGather","int32")),n=w(e,"paramsDenseValues","raggedGather"),u=w(s,"indices","raggedGather","int32"),o={paramsNestedSplits:r,paramsDenseValues:n,indices:u},l={outputRaggedRank:a},c=C.runKernel(Bs,o,l);return{outputNestedSplits:c.slice(0,c.length-1),outputDenseValues:c[c.length-1]}}const Gi=O({raggedGather_:Wi});/**
 * @license
 * Copyright 2022 Google LLC.
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
 */function Qi(t,e,s){const a=w(t,"starts","raggedRange"),r=w(e,"limits","raggedRange",a.dtype),n=w(s,"deltas","raggedRange",a.dtype),u={starts:a,limits:r,deltas:n},o=C.runKernel(qs,u);return{rtNestedSplits:o[0],rtDenseValues:o[1]}}const Ki=O({raggedRange_:Qi});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
 */function Ji(t,e,s,a,r){const n=w(t,"shape","raggedTensorToTensor","int32"),u=w(e,"values","raggedTensorToTensor"),o=w(s,"defaultValue","raggedTensorToTensor",u.dtype),l=a.map((m,h)=>w(m,`tensors${h}`,"raggedTensorToTensor","int32")),c={shape:n,values:u,defaultValue:o,rowPartitionTensors:l},p={rowPartitionTypes:r};return C.runKernel(Us,c,p)}const Xi=O({raggedTensorToTensor_:Ji});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function Yi(t,e,s){Pe(t);const a=ce(t);let r=null;if(s==null||s==="float32")r=new Float32Array(a);else if(s==="int32")r=new Int32Array(a);else if(s==="bool")r=new Uint8Array(a);else throw new Error(`Unknown data type ${s}`);for(let n=0;n<a;n++)r[n]=e();return C.makeTensor(r,t,s)}const Zi=O({rand_:Yi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function Mi(t,e,s=1,a="float32",r){if(Pe(t),s==null&&(s=1),a==null&&(a="float32"),a!=="float32"&&a!=="int32")throw new Error(`Unsupported data type ${a}`);const n=new Hs(e,s,a,r),u=Qt(t,a);for(let o=0;o<u.values.length;o++)u.values[o]=n.nextValue();return u.toTensor()}const eo=O({randomGamma_:Mi});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
 */function to(t,e,s){if(e!=null&&e==="bool")throw new Error(`Unsupported data type ${e}`);return Kt(t,0,1,e,s)}const so=O({randomStandardNormal_:to});/**
 * @license
 * Copyright 2023 Google LLC.
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
 */function ao(t,e,s,a){return Jt(t,e,s,"int32",a)}const ro=O({randomUniformInt_:ao});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function no(t){const e=w(t,"x","reverse");return v(e.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${e.rank}.`),ye(e,0)}const io=O({reverse1d_:no});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function oo(t,e){const s=w(t,"x","reverse");return v(s.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${s.rank}.`),ye(s,e)}const uo=O({reverse2d_:oo});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function lo(t,e){const s=w(t,"x","reverse");return v(s.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${s.rank}.`),ye(s,e)}const co=O({reverse3d_:lo});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function po(t,e){const s=w(t,"x","reverse");return v(s.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${s.rank}.`),ye(s,e)}const mo=O({reverse4d_:po});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
 */async function ho(t,e){const s=w(t,"x","setdiff1d"),a=w(e,"y","setdiff1d");v(s.dtype===a.dtype,()=>`x and y should have the same dtype, but got x (${s.dtype}) and y (${a.dtype}).`),v(s.rank===1,()=>`x should be 1D tensor, but got x (${s.shape}).`),v(a.rank===1,()=>`y should be 1D tensor, but got y (${a.shape}).`);const r=await s.data(),n=await a.data(),u=new Set(n);let o=0;for(let p=0;p<r.length;p++)u.has(r[p])||o++;const l=new Tt([o],s.dtype),c=new Tt([o],"int32");for(let p=0,m=0;p<r.length;p++)u.has(r[p])||(l.values[m]=r[p],c.values[m]=p,m++);return[l.toTensor(),c.toTensor()]}const fo=ho;/**
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
 */function yo(t,e,s){if(rt(t),e!=null&&e.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const a=nt(t,s);if(a.length!==4&&a.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(a.length===1&&e==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return it(t,e,a,s)}/**
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
 */function go(t,e,s){if(rt(t),e!=null&&e.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const a=nt(t,s);if(a.length!==5&&a.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(a.length===1&&e==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return it(t,e,a,s)}/**
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
 */function bo(t,e,s){if(rt(t),e!=null&&e.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const a=nt(t,s);if(a.length!==6&&a.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(a.length===1&&e==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return e=e||a,it(t,e,a,s)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
 */function No(t,e,s){const a=w(t,"tensor","tensorScatterupdate"),r=w(e,"indices","tensorScatterupdate","int32"),n=w(s,"updates","tensorScatterupdate");if(Xt(n,r,a.shape),a.dtype!==n.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${a.dtype} and ${n.dtype}.`);const u={tensor:a,indices:r,updates:n},o={};return C.runKernel(Ws,u,o)}const wo=O({tensorScatterUpdate_:No});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
 */function To(t,e){return ct(t,e,"right")}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */async function So(t){const e=w(t,"condition","whereAsync","bool"),s=await e.data(),a=Gs(e.shape,s);return t!==e&&e.dispose(),a}const ls=So;/**
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
 */async function vo(t,e,s){const a=w(t,"tensor","boolMask"),r=w(e,"mask","boolMask","bool"),n=s??0,u=r.rank,o=a.shape;v(u>0,()=>"mask cannot be scalar"),Yt(o.slice(n,n+u),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let l=1;for(let y=n;y<n+u;y++)l*=o[y];const c=o.slice(0,n).concat([l],o.slice(n+u)),p=E(a,c),m=E(r,[-1]),h=await ls(m),d=Zt(h,[1]),b=Mt(p,d,n);return t!==a&&a.dispose(),e!==r&&r.dispose(),d.dispose(),p.dispose(),m.dispose(),h.dispose(),b}const Oo=vo;/**
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
 */function ko(t,e,s,a,r=!0){const n=w(t,"v","movingAverage"),u=w(e,"x","movingAverage"),o=w(s,"decay","movingAverage");Qs(n,u),v(le(n.shape,u.shape),()=>"Shape mismatch in v and x");const l=Q(1),c=Oe(l,o);let p=ue(Oe(u,n),c);if(r){v(a!=null,()=>"When using zeroDebias: true, step is required.");const m=w(a,"step","movingAverage");p=es(p,Oe(l,ts(o,m)))}return re(n,p)}const _o=O({movingAverage_:ko});/**
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
 */function Eo(t,e,s){Pe(s);const a=w(t,"indices","scatterND","int32"),r=w(e,"updates","scatterND");Xt(r,a,s);const n={indices:a,updates:r},u={shape:s};return C.runKernel(Ks,n,u)}const Ao=O({scatterND_:Eo});function Io(t,e,s,a){if(t.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${t.shape}.`);const r=t.rank>0?t.shape[0]:1,n=t.rank>1?t.shape[1]:1;if(s.length!==n)throw new Error(`outputShape has incorrect number of elements:, ${s.length}, should be: ${n}.`);const u=e.size;if(!(e.rank===0||e.rank===1&&u===r))throw new Error(`sparseValues has incorrect shape ${e.shape}, should be [] or [${r}]`);if(e.dtype!==a.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
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
 */function Co(t,e,s,a=0){Pe(s);const r=w(t,"sparseIndices","sparseToDense","int32"),n=w(e,"sparseValues","sparseToDense","string_or_numeric"),u=w(a,"defaultValue","sparseToDense",n.dtype);Io(r,n,s,u);const o={sparseIndices:r,sparseValues:n,defaultValue:u},l={outputShape:s};return C.runKernel(Js,o,l)}const Do=O({sparseToDense_:Co});/**
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
 */function Po(t,e){const s=w(e,"indices","gatherND","int32"),r={params:w(t,"x","gatherND","string_or_numeric"),indices:s};return C.runKernel(Xs,r)}const jo=O({gatherND_:Po});/**
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
 */async function xo(t,e,s=1){const a=w(t,"predictions","inTopK"),r=w(e,"targets","inTopK");v(a.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${a.rank}`),v(a.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${a.rank} and targets rank ${r.rank}`),Yt(a.shape.slice(0,a.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const n=a.shape[a.shape.length-1];v(s>0&&s<=n,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${n}), but got ${s}`);const u=await a.data(),o=await r.data(),[l,c]=[u.length/n,n],p=Ys("bool",l);for(let m=0;m<l;m++){const h=m*c,d=u.subarray(h,h+c),b=[];for(let y=0;y<d.length;y++)b.push({value:d[y],index:y});b.sort((y,f)=>f.value-y.value),p[m]=0;for(let y=0;y<s;y++)if(b[y].index===o[m]){p[m]=1;break}}return t!==a&&a.dispose(),e!==r&&r.dispose(),ne(p,r.shape,"bool")}const $o=xo;/**
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
 */function Fo({x:t,filter:e,strides:s,pad:a,dataFormat:r="NHWC",dilations:n=[1,1],dimRoundingMode:u,bias:o,activation:l="linear",preluActivationWeights:c,leakyreluAlpha:p}){if(Zs(C.state.gradientDepth,l)===!1){let _=ss(t,e,s,a,r,n,u);return o!=null&&(_=re(_,o)),Ms(_,l,c,p)}const m=w(t,"x","depthwiseConv2d","float32"),h=w(e,"filter","depthwiseConv2d","float32");let d=m,b=!1;m.rank===3&&(b=!0,d=E(m,[1,m.shape[0],m.shape[1],m.shape[2]])),v(d.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${d.rank}.`),v(h.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${h.rank}.`),v(d.shape[3]===h.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${d.shape[3]}) must match the inChannels dimension in filter ${h.shape[2]}.`),n==null&&(n=[1,1]),v(ea(s,n),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${s} and dilations '${n}'`),ta("fused depthwiseConv2d",a,u);const y=sa(d.shape,h.shape,s,n,a,u,!0);let f;o!=null&&(f=w(o,"bias","fused conv2d"),[f]=aa(f,m),ra(y.outShape,f.shape));let N;c!=null&&(N=w(c,"prelu weights","fused depthwiseConv2d"));const S=(_,j)=>{v(na(n),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${n}'`);const[A,I,$,F]=j,K=ia(_,$,l),bt=oa(I.shape,K,A,s,a,n,u),Nt=ua(I,K,A.shape,s,a,n,u);if(F!=null){const Is=la(f,K);return[bt,Nt,Is]}return[bt,Nt]},k={x:d,filter:h,bias:f,preluActivationWeights:N},T={strides:s,pad:a,dataFormat:r,dilations:n,dimRoundingMode:u,activation:l,leakyreluAlpha:p};return o==null?St((j,A,I)=>{let $=C.runKernel(vt,k,T);return I([A,j,$]),b&&($=E($,[$.shape[1],$.shape[2],$.shape[3]])),{value:$,gradFunc:S}})(d,h):St((j,A,I,$)=>{let F=C.runKernel(vt,k,T);return $([A,j,F,I]),b&&(F=E(F,[F.shape[1],F.shape[2],F.shape[3]])),{value:F,gradFunc:S}})(d,h,f)}const zo=O({fusedDepthwiseConv2d_:Fo});/**
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
 */const Ro=Object.freeze(Object.defineProperty({__proto__:null,conv2d:ca,depthwiseConv2d:zo,matMul:pa},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Lo="model",Vo=".json",Bo=".weights.bin";function kt(t){return new Promise(e=>setTimeout(e)).then(t)}class te{constructor(e){if(!X().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(te.URL_SCHEME)&&(e=e.slice(te.URL_SCHEME.length)),(e==null||e.length===0)&&(e=Lo),this.modelJsonFileName=e+Vo,this.weightDataFileName=e+Bo}async save(e){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const s=je.join(e.weightData),a=window.URL.createObjectURL(new Blob([s],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const r=[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}],n=as(e,r),u=window.URL.createObjectURL(new Blob([JSON.stringify(n)],{type:"application/json"})),o=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(o.download=this.modelJsonFileName,o.href=u,await kt(()=>o.dispatchEvent(new MouseEvent("click"))),e.weightData!=null){const l=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;l.download=this.weightDataFileName,l.href=a,await kt(()=>l.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:lt(e)}}}}te.URL_SCHEME="downloads://";class qo{constructor(e){if(e==null||e.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,s)=>{const a=new FileReader;a.onload=r=>{const n=JSON.parse(r.target.result),u=n.modelTopology;if(u==null){s(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(n.weightsManifest==null){s(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){e({modelTopology:u});return}const l=ot(n,c=>this.loadWeights(c));e(l)},a.onerror=r=>s(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),a.readAsText(this.jsonFile)})}loadWeights(e){const s=[],a=[];for(const u of e)s.push(...u.weights),a.push(...u.paths);const r=this.checkManifestAndWeightFiles(e),n=a.map(u=>this.loadWeightsFile(u,r[u]));return Promise.all(n).then(u=>[s,u])}loadWeightsFile(e,s){return new Promise((a,r)=>{const n=new FileReader;n.onload=u=>{const o=u.target.result;a(o)},n.onerror=u=>r(`Failed to weights data from file of path '${e}'.`),n.readAsArrayBuffer(s)})}checkManifestAndWeightFiles(e){const s=[],a=this.weightsFiles.map(n=>Ot(n.name)),r={};for(const n of e)n.paths.forEach(u=>{const o=Ot(u);if(s.indexOf(o)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${o}'`);if(s.push(o),a.indexOf(o)===-1)throw new Error(`Weight file with basename '${o}' is not provided.`);r[u]=this.weightsFiles[a.indexOf(o)]});if(s.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${s.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const Uo=t=>X().getBool("IS_BROWSER")&&!Array.isArray(t)&&t.startsWith(te.URL_SCHEME)?Ho(t.slice(te.URL_SCHEME.length)):null;ut.registerSaveRouter(Uo);function Ho(t="model"){return new te(t)}function Wo(t){return new qo(t)}/**
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
 */function _t(t,e,s,a){u(t),s=s??0,a=a??1,o(s,a);let r=0;const n=l=>(l.then(c=>{const p=s+ ++r/t.length*(a-s);return e(p),c}),l);function u(l){v(l!=null&&Array.isArray(l)&&l.length>0,()=>"promises must be a none empty array")}function o(l,c){v(l>=0&&l<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${l}`),v(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${c}`),v(c>=l,()=>`startFraction must be no more than endFraction, but got startFraction ${l} and endFraction ${c}`)}return Promise.all(t.map(n))}/**
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
 */async function cs(t,e){e==null&&(e={});const s=e.fetchFunc==null?X().platform.fetch:e.fetchFunc,a=t.map(m=>s(m,e.requestInit,{isBinary:!0})),o=(e.onProgress==null?await Promise.all(a):await _t(a,e.onProgress,0,.5)).map(m=>m.arrayBuffer());return e.onProgress==null?await Promise.all(o):await _t(o,e.onProgress,.5,1)}function Go(t,e){var s;const a=e.fetchFunc==null?X().platform.fetch:e.fetchFunc;let r=0,n;return(s=e.onProgress)===null||s===void 0||s.call(e,0),new ReadableStream({pull:async u=>{for(var o;r<t.length;){n||(n=(await a(t[r],e.requestInit,{isBinary:!0})).body.getReader());const{done:l,value:c}=await n.read();if(l){r++,n=void 0,(o=e.onProgress)===null||o===void 0||o.call(e,r/t.length);continue}u.enqueue(c);return}u.close()}})}async function Qo(t,e="",s,a){return ps(u=>cs(u,{requestInit:a}))(t,e,s)}function ps(t){return async(e,s="",a)=>{const r=e.map(()=>!1),n={},u=a!=null?a.map(()=>!1):[],o=[];if(e.forEach((d,b)=>{let y=0;d.weights.forEach(f=>{const N="quantization"in f?f.quantization.dtype:f.dtype,S=ma[N]*ce(f.shape),k=()=>{r[b]=!0,n[b]==null&&(n[b]=[]),n[b].push({manifestEntry:f,groupOffset:y,sizeBytes:S})};a!=null?a.forEach((T,_)=>{T===f.name&&(k(),u[_]=!0)}):k(),o.push(f.name),y+=S})}),!u.every(d=>d)){const d=a.filter((b,y)=>!u[y]);throw new Error(`Could not find weights in manifest with names: ${d.join(", ")}. 
Manifest JSON has weights with names: ${o.join(", ")}.`)}const l=r.reduce((d,b,y)=>(b&&d.push(y),d),[]),c=[];l.forEach(d=>{e[d].paths.forEach(b=>{const y=s+(s.endsWith("/")?"":"/")+b;c.push(y)})});const p=await t(c),m={};let h=0;return l.forEach(d=>{const b=e[d].paths.length,y=new je(p.slice(h,h+b));n[d].forEach(N=>{const S=y.slice(N.groupOffset,N.groupOffset+N.sizeBytes),k=rs(S,[N.manifestEntry]);for(const T in k)m[T]=k[T]}),h+=b}),m}}/**
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
 */const Ko="application/octet-stream",Jo="application/json";class pt{constructor(e,s){if(this.DEFAULT_METHOD="POST",s==null&&(s={}),this.weightPathPrefix=s.weightPathPrefix,this.weightUrlConverter=s.weightUrlConverter,s.fetchFunc!=null?(v(typeof s.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=s.fetchFunc):this.fetch=X().platform.fetch,v(e!=null&&e.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(e)&&v(e.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,s.requestInit!=null&&s.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=s.requestInit||{},this.loadOptions=s}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const s=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);s.body=new FormData;const a=[{paths:["./model.weights.bin"],weights:e.weightSpecs}],r=as(e,a);if(s.body.append("model.json",new Blob([JSON.stringify(r)],{type:Jo}),"model.json"),e.weightData!=null){const u=je.join(e.weightData);s.body.append("model.weights.bin",new Blob([u],{type:Ko}),"model.weights.bin")}const n=await this.fetch(this.path,s);if(n.ok)return{modelArtifactsInfo:lt(e),responses:[n]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${n.status}.`)}async loadModelJSON(){const e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw new Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let s;try{s=await e.json()}catch{let u=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?u+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":u+=" Please make sure the server is serving valid JSON for this request.",new Error(u)}const a=s.modelTopology,r=s.weightsManifest;if(a==null&&r==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return s}async load(){if(this.loadOptions.streamWeights)return this.loadStream();const e=await this.loadModelJSON();return ot(e,s=>this.loadWeights(s))}async loadStream(){const e=await this.loadModelJSON(),s=await this.getWeightUrls(e.weightsManifest),a=Re(e.weightsManifest),r=()=>Go(s,this.loadOptions);return Object.assign(Object.assign({},e),{weightSpecs:a,getWeightStream:r})}async getWeightUrls(e){const s=Array.isArray(this.path)?this.path[1]:this.path,[a,r]=Xo(s),n=this.weightPathPrefix||a,u=[],o=[];for(const l of e)for(const c of l.paths)this.weightUrlConverter!=null?o.push(this.weightUrlConverter(c)):u.push(n+c+r);return this.weightUrlConverter&&u.push(...await Promise.all(o)),u}async loadWeights(e){const s=await this.getWeightUrls(e),a=Re(e),r=await cs(s,this.loadOptions);return[a,r]}}pt.URL_SCHEME_REGEX=/^https?:\/\//;function Xo(t){const e=t.lastIndexOf("/"),s=t.lastIndexOf("?"),a=t.substring(0,e),r=s>e?t.substring(s):"";return[a+"/",r]}function Le(t){return t.match(pt.URL_SCHEME_REGEX)!=null}const ms=(t,e)=>{if(typeof fetch>"u"&&(e==null||e.fetchFunc==null))return null;{let s=!0;if(Array.isArray(t)?s=t.every(a=>Le(a)):s=Le(t),s)return mt(t,e)}return null};ut.registerSaveRouter(ms);ut.registerLoadRouter(ms);function mt(t,e){return new pt(t,e)}function Yo(t,e){return mt(t,e)}/**
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
 */class $e{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class hs{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class Zo{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=s=>Promise.resolve(e.save(s)))}}function Mo(t,e,s,a){const r=arguments;return new Zo(ds(...r))}function ds(t,e,s,a){return arguments.length===1?t.modelTopology!=null||t.weightSpecs!=null?new $e(t):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new $e({modelTopology:t})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new $e({modelTopology:t,weightSpecs:e,weightData:s,trainingConfig:a}))}function eu(t){return new hs(t)}function tu(t){return new hs(t)}/**
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
 */const fs=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:je,browserFiles:Wo,browserHTTPRequest:Yo,concatenateArrayBuffers:ha,copyModel:da,decodeWeights:rs,decodeWeightsStream:ns,encodeWeights:fa,fromMemory:Mo,fromMemorySync:ds,getLoadHandlers:ya,getModelArtifactsForJSON:ot,getModelArtifactsForJSONSync:ga,getModelArtifactsInfoForJSON:lt,getSaveHandlers:ba,getWeightSpecs:Re,http:mt,isHTTPScheme:Le,listModels:Na,loadWeights:Qo,moveModel:wa,registerLoadRouter:Ta,registerSaveRouter:Sa,removeModel:va,weightsLoaderFactory:ps,withSaveHandler:eu,withSaveHandlerSync:tu},Symbol.toStringTag,{value:"Module"}));/**
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
 */const su={};function ys(t){return su[t]}/**
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
 */function i(t,e,s,a,r){const n=e.inputParams[t];if(n&&n.inputIndexStart!==void 0){const o=n.inputIndexStart,l=n.inputIndexEnd===0?void 0:n.inputIndexEnd===void 0?o+1:n.inputIndexEnd,c=o<0?e.inputNames.length+o:o;if(n.type==="tensor")return D(e.inputNames[c],s,a,r);if(n.type==="tensors"){const h=e.inputs.slice(o,l);return e.inputNames.slice(o,l).filter((b,y)=>{var f;return((f=h[y])===null||f===void 0?void 0:f.op)!=="NoOp"}).map(b=>D(b,s,a,r))}const p=D(e.inputNames[c],s,a,r),m=p.dataSync();return n.type==="number"?m[0]:Oa(p.shape,m)}const u=e.attrParams[t];return u&&u.value}function D(t,e,s,a){const[r,n]=z(t,s);if(a!=null){const o=a.getHashTableHandleByName(r);if(o!=null)return o}const u=s.currentContextIds.find(o=>!!e[Ae(r,o)]);return u!==void 0?e[Ae(r,u)][n]:void 0}function Et(t,e,s){return e[Ae(t,s.currentContextId)]}function H(t,e){const[s,a,r]=z(t,e);return[Ae(s,e&&e.currentContextId),a,r]}function Ae(t,e){return e?`${t}-${e}`:t}function z(t,e){if(t==="")return["",0,void 0];const s=e!=null&&e.parseNodeNameCache!=null;if(s){const n=e.parseNodeNameCache.get(t);if(n!=null)return n}const a=t.split(":");let r;if(a.length===1)r=[t,0,void 0];else{const n=a[0],u=a.length===3?a[1]:void 0,o=Number(a[a.length-1]);r=[n,o,u]}return s&&e.parseNodeNameCache.set(t,r),r}function ke(t,e,s){let a=i("pad",t,e,s);if(a==="explicit"){a=i("explicitPaddings",t,e,s);const r=[[0,0],[0,0],[0,0],[0,0]];for(let n=0;n<4;n++)r[n][0]=a[n*2],r[n][1]=a[n*2+1];return r}return a}function W(t){return t.kept?t:is(t)}/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const au=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],ru=Object.freeze(Object.defineProperty({__proto__:null,json:au},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const nu=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],iu=Object.freeze(Object.defineProperty({__proto__:null,json:nu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const ou=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],uu=Object.freeze(Object.defineProperty({__proto__:null,json:ou},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const lu=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],cu=Object.freeze(Object.defineProperty({__proto__:null,json:lu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const pu=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniformInt",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number"},{tfName:"maxval",name:"maxval",type:"number"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],mu=Object.freeze(Object.defineProperty({__proto__:null,json:pu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const hu=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],du=Object.freeze(Object.defineProperty({__proto__:null,json:hu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const fu=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],yu=Object.freeze(Object.defineProperty({__proto__:null,json:fu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const gu=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],bu=Object.freeze(Object.defineProperty({__proto__:null,json:gu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Nu=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],wu=Object.freeze(Object.defineProperty({__proto__:null,json:Nu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Tu=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],Su=Object.freeze(Object.defineProperty({__proto__:null,json:Tu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const vu=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BitwiseAnd",category:"logical",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}]}],Ou=Object.freeze(Object.defineProperty({__proto__:null,json:vu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const ku=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}],_u=Object.freeze(Object.defineProperty({__proto__:null,json:ku},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Eu=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]}],Au=Object.freeze(Object.defineProperty({__proto__:null,json:Eu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Iu=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],Cu=Object.freeze(Object.defineProperty({__proto__:null,json:Iu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Du=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],Pu=Object.freeze(Object.defineProperty({__proto__:null,json:Du},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const ju=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],xu=Object.freeze(Object.defineProperty({__proto__:null,json:ju},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const $u=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],Fu=Object.freeze(Object.defineProperty({__proto__:null,json:$u},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const zu=[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],Ru=Object.freeze(Object.defineProperty({__proto__:null,json:zu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Lu=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"EnsureShape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],Vu=Object.freeze(Object.defineProperty({__proto__:null,json:Lu},Symbol.toStringTag,{value:"Module"}));/**
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
 */class At{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const e=[ru,iu,uu,cu,mu,du,yu,bu,wu,Su,Ou,_u,Au,Cu,Pu,xu,Fu,Ru,Vu],s=[].concat(...e.map(a=>a.json));this.opMappers=s.reduce((a,r)=>(a[r.tfOpName]=r,a),{})}transformGraph(e,s={}){const a=e.node,r=[],n=[],u=[],o=a.reduce((y,f)=>(y[f.name]=this.mapNode(f),f.op.startsWith("Placeholder")?r.push(y[f.name]):f.op==="Const"?n.push(y[f.name]):(f.input==null||f.input.length===0)&&u.push(y[f.name]),y),{});let l=[];const c=[];let p={},m={};s!=null&&(p=this.mapSignatureEntries(s.inputs),m=this.mapSignatureEntries(s.outputs));const h=Object.keys(o);h.forEach(y=>{const f=o[y];f.inputNames.forEach((N,S)=>{const[k,,T]=H(N),_=o[k];if(_.outputs!=null){const j=_.outputs.indexOf(T);if(j!==-1){const A=`${k}:${j}`;f.inputNames[S]=A}}f.inputs.push(_),_.children.push(f)})}),Object.keys(m).length===0?h.forEach(y=>{const f=o[y];f.children.length===0&&c.push(f)}):Object.keys(m).forEach(y=>{const[f]=H(y),N=o[f];N!=null&&(N.signatureKey=m[y],c.push(N))}),Object.keys(p).length>0?Object.keys(p).forEach(y=>{const[f]=H(y),N=o[f];N&&(N.signatureKey=p[y],l.push(N))}):l=r;let d={};e.library!=null&&e.library.function!=null&&(d=e.library.function.reduce((y,f)=>(y[f.signature.name]=this.mapFunction(f),y),{}));const b={nodes:o,inputs:l,outputs:c,weights:n,placeholders:r,signature:s,functions:d};return u.length>0&&(b.initNodes=u),b}mapSignatureEntries(e){return Object.keys(e||{}).reduce((s,a)=>(s[e[a].name]=a,s),{})}mapNode(e){const s=ys(e.op)||this.opMappers[e.op]||{};e.attr==null&&(e.attr={});const a={name:e.name,op:e.op,category:s.category,inputNames:(e.input||[]).map(r=>r.startsWith("^")?r.slice(1):r),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:s.outputs};return s.inputs!=null&&(a.inputParams=s.inputs.reduce((r,n)=>(r[n.name]={type:n.type,inputIndexStart:n.start,inputIndexEnd:n.end},r),{})),s.attrs!=null&&(a.attrParams=s.attrs.reduce((r,n)=>{const u=n.type;let o;switch(n.type){case"string":o=Ve(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=Ve(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"string[]":o=Qe(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=Qe(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"number":o=qe(e.attr,n.tfName,n.defaultValue||0),o===void 0&&n.tfDeprecatedName&&(o=qe(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"number[]":o=Ge(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=Ge(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"bool":o=Be(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=Be(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"bool[]":o=Je(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=Je(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"shape":o=We(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=We(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"shape[]":o=Ke(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=Ke(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"dtype":o=Ue(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=Ue(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"dtype[]":o=He(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=He(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"func":o=It(e.attr,n.tfName,n.defaultValue),o===void 0&&n.tfDeprecatedName&&(o=It(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${n.type} for op: ${e.op}`)}return r[n.name]={value:o,type:u},r},{})),a}mapFunction(e){const s=e.nodeDef,a=[],r=[];let n={};s!=null&&(n=s.reduce((m,h)=>(m[h.name]=this.mapNode(h),h.op==="Const"&&r.push(m[h.name]),m),{}));const u=[],o=[];e.signature.inputArg.forEach(m=>{const[h]=H(m.name),d={name:h,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:ht(m.type),type:"dtype"}},children:[]};d.signatureKey=m.name,u.push(d),n[h]=d}),Object.keys(n).forEach(m=>{const h=n[m];h.inputNames.forEach((d,b)=>{const[y,,f]=H(d),N=n[y];if(N.outputs!=null){const S=N.outputs.indexOf(f);if(S!==-1){const k=`${y}:${S}`;h.inputNames[b]=k}}h.inputs.push(N),N.children.push(h)})});const c=e.ret;e.signature.outputArg.forEach(m=>{const[h,d]=H(c[m.name]),b=n[h];b!=null&&(b.defaultOutput=d,o.push(b))});const p=this.mapArgsToSignature(e);return{nodes:n,inputs:u,outputs:o,weights:r,placeholders:a,signature:p}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((s,a)=>(s[a.name]=this.mapArgToTensorInfo(a),s),{}),outputs:e.signature.outputArg.reduce((s,a)=>(s[a.name]=this.mapArgToTensorInfo(a,e.ret),s),{})}}mapArgToTensorInfo(e,s){let a=e.name;return s!=null&&(a=s[a]),{name:a,dtype:e.type}}}function Bu(t){const e=X().global;if(typeof e.atob<"u")return e.atob(t);if(typeof Buffer<"u")return new Buffer(t,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function gs(t,e){const s=Array.isArray(t)?String.fromCharCode.apply(null,t):Bu(t);return e?s:s.toLowerCase()}function Ve(t,e,s,a=!1){const r=t[e];return r!=null?gs(r.s,a):s}function Be(t,e,s){const a=t[e];return a?a.b:s}function qe(t,e,s){const a=t[e]||{},r=a.i!=null?a.i:a.f!=null?a.f:s;return typeof r=="number"?r:parseInt(r,10)}function ht(t){switch(typeof t=="string"&&(t=L[t]),t){case L.DT_FLOAT:case L.DT_HALF:return"float32";case L.DT_INT32:case L.DT_INT64:case L.DT_INT8:case L.DT_UINT8:return"int32";case L.DT_BOOL:return"bool";case L.DT_DOUBLE:return"float32";case L.DT_STRING:return"string";case L.DT_COMPLEX64:case L.DT_COMPLEX128:return"complex64";default:return null}}function It(t,e,s){const a=t[e];return a&&a.func?a.func.name:s}function Ue(t,e,s){const a=t[e];return a&&a.type?ht(a.type):s}function He(t,e,s){const a=t[e];return a&&a.list&&a.list.type?a.list.type.map(r=>ht(r)):s}function bs(t){if(!t.unknownRank)return t.dim!=null?t.dim.map(e=>typeof e.size=="number"?e.size:parseInt(e.size,10)):[]}function We(t,e,s){const a=t[e];return a&&a.shape?bs(a.shape):s}function Ge(t,e,s){const a=t[e];return a?((a.list.f&&a.list.f.length?a.list.f:a.list.i)||[]).map(r=>typeof r=="number"?r:parseInt(r,10)):s}function Qe(t,e,s,a=!1){const r=t[e];return r&&r.list&&r.list.s?r.list.s.map(n=>gs(n,a)):s}function Ke(t,e,s){const a=t[e];return a&&a.list&&a.list.shape?a.list.shape.map(r=>bs(r)):s}function Je(t,e,s){const a=t[e];return a&&a.list&&a.list.b?a.list.b:s}/**
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
 */class qu{constructor(e,s,a){this.node=e,this.tensorMap=s,this.context=a,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(r=>this.getInput(r)),e.rawAttrs!=null&&(this.attrs=Object.keys(e.rawAttrs).reduce((r,n)=>(r[n]=this.getAttr(n),r),{}))}getInput(e){return D(e,this.tensorMap,this.context)}getAttr(e,s){const a=this.node.rawAttrs[e];if(a.tensor!=null)return D(e,this.tensorMap,this.context);if(a.i!=null||a.f!=null)return qe(this.node.rawAttrs,e,s);if(a.s!=null)return Ve(this.node.rawAttrs,e,s);if(a.b!=null)return Be(this.node.rawAttrs,e,s);if(a.shape!=null)return We(this.node.rawAttrs,e,s);if(a.type!=null)return Ue(this.node.rawAttrs,e,s);if(a.list!=null){if(a.list.i!=null||a.list.f!=null)return Ge(this.node.rawAttrs,e,s);if(a.list.s!=null)return Qe(this.node.rawAttrs,e,s);if(a.list.shape!=null)return Ke(this.node.rawAttrs,e,s);if(a.list.b!=null)return Je(this.node.rawAttrs,e,s);if(a.list.type!=null)return He(this.node.rawAttrs,e,s)}return s}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */const P=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:ka,abs:_a,acos:Ea,acosh:Aa,add:re,addN:di,all:Ia,any:Ca,argMax:Da,argMin:Pa,asin:ja,asinh:xa,atan:$a,atan2:Fa,atanh:za,avgPool:Ra,avgPool3d:La,basicLSTMCell:yi,batchNorm:Va,batchNorm2d:Ba,batchNorm3d:qa,batchNorm4d:Ua,batchToSpaceND:Ha,bincount:Wa,bitwiseAnd:bi,booleanMaskAsync:Oo,broadcastArgs:wi,broadcastTo:Ga,buffer:Qt,cast:Qa,ceil:Ka,clipByValue:Ja,clone:is,complex:Xa,concat:De,concat1d:Ya,concat2d:Za,concat3d:Ma,concat4d:er,conv1d:tr,conv2d:sr,conv2dTranspose:ar,conv3d:rr,conv3dTranspose:nr,cos:ir,cosh:or,cosineWindow:ur,cumprod:lr,cumsum:cr,denseBincount:pr,depthToSpace:mr,depthwiseConv2d:ss,diag:Si,dilation2d:hr,div:es,divNoNan:dr,dot:fr,dropout:yr,einsum:gr,elu:br,enclosingPowerOfTwo:Nr,ensureShape:Oi,equal:wr,erf:Tr,euclideanNorm:Sr,exp:vr,expandDims:Or,expm1:kr,eye:_r,fft:Er,fill:Ar,floor:Ir,floorDiv:Cr,fused:Ro,gather:Mt,gatherND:jo,greater:Dr,greaterEqual:Pr,ifft:jr,imag:xr,image:os,inTopKAsync:$o,irfft:$r,isFinite:Fr,isInf:zr,isNaN:Rr,leakyRelu:Lr,less:Vr,lessEqual:Br,linalg:qr,linspace:ki,localResponseNormalization:Ur,log:Hr,log1p:Wr,logSigmoid:Gr,logSoftmax:Qr,logSumExp:Kr,logicalAnd:Jr,logicalNot:Xr,logicalOr:Yr,logicalXor:Zr,losses:Mr,lowerBound:Ei,matMul:Z,max:en,maxPool:tn,maxPool3d:sn,maxPoolWithArgmax:Ii,maximum:an,mean:rn,meshgrid:Ci,min:nn,minimum:on,mirrorPad:un,mod:ln,moments:cn,movingAverage:_o,mul:ue,multiRNNCell:Pi,multinomial:xi,neg:pn,norm:mn,notEqual:hn,oneHot:dn,ones:oe,onesLike:fn,op:O,outerProduct:Fi,pad:fe,pad1d:Ri,pad2d:Vi,pad3d:qi,pad4d:Hi,pool:yn,pow:ts,prelu:gn,print:bn,prod:Nn,raggedGather:Gi,raggedRange:Ki,raggedTensorToTensor:Xi,rand:Zi,randomGamma:eo,randomNormal:Kt,randomStandardNormal:so,randomUniform:Jt,randomUniformInt:ro,range:wn,real:Tn,reciprocal:Sn,relu:vn,relu6:On,reshape:E,reverse:ye,reverse1d:io,reverse2d:uo,reverse3d:co,reverse4d:mo,rfft:kn,round:_n,rsqrt:En,scalar:Q,scatterND:Ao,searchSorted:ct,selu:An,separableConv2d:In,setdiff1dAsync:fo,sigmoid:ve,sign:Cn,signal:Dn,sin:Pn,sinh:jn,slice:M,slice1d:xn,slice2d:$n,slice3d:Fn,slice4d:zn,softmax:Rn,softplus:Ln,spaceToBatchND:Vn,sparse:Bn,sparseToDense:Do,spectral:qn,split:Un,sqrt:Hn,square:Wn,squaredDifference:Gn,squeeze:Zt,stack:me,step:Qn,stridedSlice:Kn,string:Jn,sub:Oe,sum:Xn,tan:Yn,tanh:ze,tensor:ne,tensor1d:Zn,tensor2d:Mn,tensor3d:pi,tensor4d:yo,tensor5d:go,tensor6d:bo,tensorScatterUpdate:wo,tile:ei,topk:us,transpose:ti,truncatedNormal:si,unique:ai,unsortedSegmentSum:ri,unstack:ge,upperBound:To,variable:ni,where:ii,whereAsync:ls,zeros:oi,zerosLike:ui},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Uu=(t,e,s,a=P)=>{switch(t.op){case"BiasAdd":case"AddV2":case"Add":return[a.add(i("a",t,e,s),i("b",t,e,s))];case"AddN":return[a.addN(i("tensors",t,e,s))];case"FloorMod":case"Mod":return[a.mod(i("a",t,e,s),i("b",t,e,s))];case"Mul":return[a.mul(i("a",t,e,s),i("b",t,e,s))];case"RealDiv":case"Div":return[a.div(i("a",t,e,s),i("b",t,e,s))];case"DivNoNan":return[a.divNoNan(i("a",t,e,s),i("b",t,e,s))];case"FloorDiv":return[a.floorDiv(i("a",t,e,s),i("b",t,e,s))];case"Sub":return[a.sub(i("a",t,e,s),i("b",t,e,s))];case"Minimum":return[a.minimum(i("a",t,e,s),i("b",t,e,s))];case"Maximum":return[a.maximum(i("a",t,e,s),i("b",t,e,s))];case"Pow":return[a.pow(i("a",t,e,s),i("b",t,e,s))];case"SquaredDifference":return[a.squaredDifference(i("a",t,e,s),i("b",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const Hu=(t,e,s,a=P)=>{switch(t.op){case"Abs":case"ComplexAbs":return[a.abs(i("x",t,e,s))];case"Acos":return[a.acos(i("x",t,e,s))];case"Acosh":return[a.acosh(i("x",t,e,s))];case"Asin":return[a.asin(i("x",t,e,s))];case"Asinh":return[a.asinh(i("x",t,e,s))];case"Atan":return[a.atan(i("x",t,e,s))];case"Atan2":return[a.atan2(i("x",t,e,s),i("y",t,e,s))];case"Atanh":return[a.atanh(i("x",t,e,s))];case"Ceil":return[a.ceil(i("x",t,e,s))];case"Complex":return[a.complex(i("real",t,e,s),i("imag",t,e,s))];case"Cos":return[a.cos(i("x",t,e,s))];case"Cosh":return[a.cosh(i("x",t,e,s))];case"Elu":return[a.elu(i("x",t,e,s))];case"Erf":return[a.erf(i("x",t,e,s))];case"Exp":return[a.exp(i("x",t,e,s))];case"Expm1":return[a.expm1(i("x",t,e,s))];case"Floor":return[a.floor(i("x",t,e,s))];case"Log":return[a.log(i("x",t,e,s))];case"Log1p":return[a.log1p(i("x",t,e,s))];case"Imag":return[a.imag(i("x",t,e,s))];case"Neg":return[a.neg(i("x",t,e,s))];case"Reciprocal":return[a.reciprocal(i("x",t,e,s))];case"Real":return[a.real(i("x",t,e,s))];case"Relu":return[a.relu(i("x",t,e,s))];case"Round":return[a.round(i("x",t,e,s))];case"Selu":return[a.selu(i("x",t,e,s))];case"Sigmoid":return[a.sigmoid(i("x",t,e,s))];case"Sin":return[a.sin(i("x",t,e,s))];case"Sign":return[a.sign(i("x",t,e,s))];case"Sinh":return[a.sinh(i("x",t,e,s))];case"Softplus":return[a.softplus(i("x",t,e,s))];case"Sqrt":return[a.sqrt(i("x",t,e,s))];case"Square":return[a.square(i("x",t,e,s))];case"Tanh":return[a.tanh(i("x",t,e,s))];case"Tan":return[a.tan(i("x",t,e,s))];case"ClipByValue":return[a.clipByValue(i("x",t,e,s),i("clipValueMin",t,e,s),i("clipValueMax",t,e,s))];case"Relu6":return[a.relu6(i("x",t,e,s))];case"Rsqrt":return[a.rsqrt(D(t.inputNames[0],e,s))];case"LeakyRelu":return[a.leakyRelu(i("x",t,e,s),i("alpha",t,e,s))];case"Prelu":return[a.prelu(i("x",t,e,s),i("alpha",t,e,s))];case"IsNan":return[a.isNaN(D(t.inputNames[0],e,s))];case"IsInf":return[a.isInf(D(t.inputNames[0],e,s))];case"IsFinite":return[a.isFinite(D(t.inputNames[0],e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */function V(t,e,s=""){if(!(typeof t=="number"||typeof e=="number")){v(t.length===e.length,()=>s+` Shapes ${t} and ${e} must match`);for(let a=0;a<t.length;a++){const r=t[a],n=e[a];v(r<0||n<0||r===n,()=>s+` Shapes ${t} and ${e} must match`)}}}function Ct(t){return!(typeof t=="number"||t.some(e=>e<0))}function ie(t,e,s){let a=Xe(t,s);const r=!Ct(a);if(r&&e.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${a}`);if(r&&e.forEach(n=>{a=Xe(n.shape,a)}),!Ct(a))throw new Error(`Non-fully-defined elementShape: ${a}`);return a}function Xe(t,e){if(typeof t=="number")return e;if(typeof e=="number")return t;if(t.length!==e.length)throw new Error(`Incompatible ranks during merge: ${t} vs. ${e}`);const s=[];for(let a=0;a<t.length;++a){const r=t[a],n=e[a];if(r>=0&&n>=0&&r!==n)throw new Error(`Incompatible shape during merge: ${t} vs. ${e}`);s[a]=r>=0?r:n}return s}/**
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
 */class Wu{constructor(e,s,a,r,n,u,o){this.name=e,this.dtype=s,this.maxSize=a,this.elementShape=r,this.identicalElementShapes=n,this.dynamicSize=u,this.clearAfterRead=o,this.tensors=[],this.closed_=!1,this.idTensor=Q(0),G(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(s=>{(e==null||!e.has(s.tensor.id))&&s.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw new Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);const s=this.tensors[e];if(s.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(s.cleared=!0),s.read=!0,s.tensor}readMany(e){return e.map(s=>this.read(s))}write(e,s){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw new Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);const a=this.tensors[e]||{};if(s.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${s.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=s.shape),V(this.elementShape,s.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),a.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(a.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);a.tensor=s,G(s),a.written=!0,this.tensors[e]=a}writeMany(e,s){if(e.length!==s.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${s.length}.`);e.forEach((a,r)=>this.write(a,s[r]))}gather(e,s){if(s&&s!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${s}`);if(e)e=e.slice(0,this.size());else{e=[];for(let r=0;r<this.size();r++)e.push(r)}if(e.length===0)return ne([],[0].concat(this.elementShape));const a=this.readMany(e);return V(this.elementShape,a[0].shape,"TensorArray shape mismatch: "),me(a,0)}concat(e){if(e&&e!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(this.size()===0)return ne([],[0].concat(this.elementShape));const s=[];for(let r=0;r<this.size();r++)s.push(r);const a=this.readMany(s);return V(this.elementShape,a[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${a[0].shape})`),De(a,0)}scatter(e,s){if(s.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${s.dtype}`);if(e.length!==s.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${s.shape[0]}`);const a=Math.max(...e);if(!this.dynamicSize&&a>=this.maxSize)throw new Error(`Max index must be < array size (${a}  vs. ${this.maxSize})`);this.writeMany(e,ge(s,0))}split(e,s){if(s.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${s.dtype}`);let a=0;const r=e.map(l=>(a+=l,a));if(a!==s.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${a}, and tensor's shape is: ${s.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);const n=a===0?0:s.size/a,u=[];U(()=>{s=E(s,[1,a,n]);for(let l=0;l<e.length;++l){const p=[0,l===0?0:r[l-1],0],m=[1,e[l],n];u[l]=E(M(s,p,m),this.elementShape)}return u});const o=[];for(let l=0;l<e.length;l++)o[l]=l;this.writeMany(o,u)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */class se{get id(){return this.idTensor.id}constructor(e,s,a,r=-1){this.tensors=e,this.elementShape=s,this.elementDtype=a,e?.forEach(n=>{if(a!==n.dtype)throw new Error(`Invalid data types; op elements ${a}, but list elements ${n.dtype}`);V(s,n.shape,"TensorList shape mismatch: "),G(n)}),this.idTensor=Q(0),this.maxNumElements=r,G(this.idTensor)}copy(){return new se([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(s=>{(e==null||!e.has(s.id))&&s.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,s,a=-1){if(s!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${this.elementDtype}`);if(a!==-1&&this.tensors.length!==a)throw new Error(`Operation expected a list with ${a} elements but got a list with ${this.tensors.length} elements.`);V(e,this.elementShape,"TensorList shape mismatch: ");const r=ie(this.elementShape,this.tensors,e);return U(()=>{const n=this.tensors.map(u=>E(u,r));return me(n,0)})}popBack(e,s){if(s!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const a=ie(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,V(r.shape,e,"TensorList shape mismatch: "),E(r,a)}pushBack(e){if(e.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(V(e.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");G(e),this.tensors.push(e)}resize(e){if(e<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${e}`);if(this.maxNumElements!==-1&&e>this.maxNumElements)throw new Error(`TensorListResize input size ${e} is greater maxNumElement ${this.maxNumElements}.`);const s=new se([],this.elementShape,this.elementDtype,this.maxNumElements);s.tensors.length=e;for(let a=0;a<Math.min(this.tensors.length,e);++a)s.tensors[a]=this.tensors[a];return s}getItem(e,s,a){if(a!==this.elementDtype)throw new Error(`Invalid data types; op elements ${a}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw new Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(this.tensors[e]==null)throw new Error(`element at index ${e} is null.`);V(this.tensors[e].shape,s,"TensorList shape mismatch: ");const r=ie(this.elementShape,this.tensors,s);return E(this.tensors[e],r)}setItem(e,s){if(s.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s.dtype}, but list elements ${this.elementDtype}`);if(e<0||this.maxNumElements!==-1&&e>=this.maxNumElements)throw new Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);V(this.elementShape,s.shape,"TensorList shape mismatch: "),G(s),this.tensors[e]!=null&&(this.tensors[e].kept=!1),this.tensors[e]=s}gather(e,s,a){if(s!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${this.elementDtype}`);V(this.elementShape,a,"TensorList shape mismatch: "),e=e.slice(0,this.size());const r=ie(this.elementShape,this.tensors,a);return e.length===0?ne([],[0].concat(r)):U(()=>{const n=e.map(u=>E(this.tensors[u],r));return me(n,0)})}concat(e,s){if(e&&e!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);V(this.elementShape,s,"TensorList shape mismatch: ");const a=ie(this.elementShape,this.tensors,s);return this.size()===0?ne([],[0].concat(a)):U(()=>{const r=this.tensors.map(n=>E(n,a));return De(r,0)})}}function Gu(t,e,s){const a=t.dtype;if(t.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${t.shape}`);if(t.dtype!==s)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${s}`);const r=t.shape.slice(1);V(r,e,"TensorList shape mismatch: ");const n=ge(t);return new se(n,e,a)}function Qu(t,e,s,a){return new se([],t,e,a)}function Ku(t,e,s,a){if(e.length!==t.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);const r=Math.max(...e);if(a!=null&&a!==-1&&r>=a)throw new Error(`Max index must be < array size (${r}  vs. ${a})`);const n=new se([],s,t.dtype,a),u=ge(t,0);return e.forEach((o,l)=>{n.setItem(o,u[l])}),n}function Ju(t,e,s){let a=0;const r=e.map(p=>(a+=p,a));if(a!==t.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${a}, and tensor's shape is: ${t.shape}`);const n=t.shape.slice(1),u=Xe(n,s),o=a===0?0:t.size/a,l=U(()=>{const p=[];t=E(t,[1,a,o]);for(let m=0;m<e.length;++m){const d=[0,m===0?0:r[m-1],0],b=[1,e[m],o];p[m]=E(M(t,d,b),u)}return t.dispose(),p}),c=new se([],s,t.dtype,e.length);for(let p=0;p<l.length;p++)c.setItem(p,l[p]);return c}/**
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
 */const Xu=async(t,e,s)=>{switch(t.op){case"If":case"StatelessIf":{const a=i("thenBranch",t,e,s),r=i("elseBranch",t,e,s),n=i("cond",t,e,s),u=i("args",t,e,s);return(await n.data())[0]?s.functionMap[a].executeFunctionAsync(u,s.tensorArrayMap,s.tensorListMap):s.functionMap[r].executeFunctionAsync(u,s.tensorArrayMap,s.tensorListMap)}case"While":case"StatelessWhile":{const a=i("body",t,e,s),r=i("cond",t,e,s),n=i("args",t,e,s),u=await s.functionMap[r].executeFunctionAsync(n,s.tensorArrayMap,s.tensorListMap),o=n.map(p=>p.id);let l=await u[0].data();u.forEach(p=>{!p.kept&&o.indexOf(p.id)===-1&&p.dispose()});let c=n;for(;l[0];){const p=c;c=await s.functionMap[a].executeFunctionAsync(c,s.tensorArrayMap,s.tensorListMap);const m=c.map(d=>d.id);p.forEach(d=>{!d.kept&&o.indexOf(d.id)===-1&&m.indexOf(d.id)===-1&&d.dispose()});const h=await s.functionMap[r].executeFunctionAsync(c,s.tensorArrayMap,s.tensorListMap);l=await h[0].data(),h.forEach(d=>{!d.kept&&o.indexOf(d.id)===-1&&m.indexOf(d.id)===-1&&d.dispose()})}return c}case"LoopCond":{const a=i("pred",t,e,s);return[W(a)]}case"Switch":{const a=i("pred",t,e,s);let r=i("data",t,e,s);return r.kept||(r=W(r)),(await a.data())[0]?[void 0,r]:[r,void 0]}case"Merge":{const a=t.inputNames.find(r=>D(r,e,s)!==void 0);if(a){const r=D(a,e,s);return[W(r)]}return}case"Enter":{const a=i("frameName",t,e,s),r=i("tensor",t,e,s);return s.enterFrame(a),[W(r)]}case"Exit":{const a=i("tensor",t,e,s);return s.exitFrame(),[W(a)]}case"NextIteration":{const a=i("tensor",t,e,s);return s.nextIteration(),[W(a)]}case"TensorArrayV3":{const a=i("size",t,e,s),r=i("dtype",t,e,s),n=i("elementShape",t,e,s),u=i("dynamicSize",t,e,s),o=i("clearAfterRead",t,e,s),l=i("identicalElementShapes",t,e,s),c=i("name",t,e,s),p=new Wu(c,r,a,n,l,u,o);return s.addTensorArray(p),[p.idTensor,Q(1)]}case"TensorArrayWriteV3":{const a=i("tensorArrayId",t,e,s),r=i("index",t,e,s),n=i("tensor",t,e,s),u=s.getTensorArray(a.id);return u.write(r,n),[u.idTensor]}case"TensorArrayReadV3":{const a=i("tensorArrayId",t,e,s),r=i("index",t,e,s);return[s.getTensorArray(a.id).read(r)]}case"TensorArrayGatherV3":{const a=i("tensorArrayId",t,e,s),r=i("indices",t,e,s),n=i("dtype",t,e,s);return[s.getTensorArray(a.id).gather(r,n)]}case"TensorArrayScatterV3":{const a=i("tensorArrayId",t,e,s),r=i("indices",t,e,s),n=i("tensor",t,e,s),u=s.getTensorArray(a.id);return u.scatter(r,n),[u.idTensor]}case"TensorArrayConcatV3":{const a=i("tensorArrayId",t,e,s),r=s.getTensorArray(a.id),n=i("dtype",t,e,s);return[r.concat(n)]}case"TensorArraySplitV3":{const a=i("tensorArrayId",t,e,s),r=i("tensor",t,e,s),n=i("lengths",t,e,s),u=s.getTensorArray(a.id);return u.split(n,r),[u.idTensor]}case"TensorArraySizeV3":{const a=i("tensorArrayId",t,e,s),r=s.getTensorArray(a.id);return[Q(r.size(),"int32")]}case"TensorArrayCloseV3":{const a=i("tensorArrayId",t,e,s),r=s.getTensorArray(a.id);return r.clearAndClose(),[r.idTensor]}case"TensorListSetItem":{const a=i("tensorListId",t,e,s),r=i("index",t,e,s),n=i("tensor",t,e,s),u=s.getTensorList(a.id);return u.setItem(r,n),[u.idTensor]}case"TensorListGetItem":{const a=i("tensorListId",t,e,s),r=i("index",t,e,s),n=i("elementShape",t,e,s),u=i("elementDType",t,e,s);return[s.getTensorList(a.id).getItem(r,n,u)]}case"TensorListScatterV2":case"TensorListScatter":{const a=i("indices",t,e,s),r=i("tensor",t,e,s),n=i("elementShape",t,e,s),u=i("numElements",t,e,s),o=Ku(r,a,n,u);return s.addTensorList(o),[o.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const a=i("elementShape",t,e,s),r=i("elementDType",t,e,s);let n;t.op==="TensorListReserve"?n="numElements":n="maxNumElements";const u=i(n,t,e,s),o=t.op==="TensorListReserve"?-1:u,l=Qu(a,r,u,o);return s.addTensorList(l),[l.idTensor]}case"TensorListGather":{const a=i("tensorListId",t,e,s),r=i("indices",t,e,s),n=i("elementShape",t,e,s),u=i("elementDType",t,e,s);return[s.getTensorList(a.id).gather(r,u,n)]}case"TensorListStack":{const a=i("tensorListId",t,e,s),r=i("elementShape",t,e,s),n=i("elementDType",t,e,s),u=i("numElements",t,e,s);return[s.getTensorList(a.id).stack(r,n,u)]}case"TensorListFromTensor":{const a=i("tensor",t,e,s),r=i("elementShape",t,e,s),n=i("elementDType",t,e,s),u=Gu(a,r,n);return s.addTensorList(u),[u.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const a=i("tensorListId",t,e,s),r=s.getTensorList(a.id),n=i("dtype",t,e,s),u=i("elementShape",t,e,s);return[r.concat(n,u)]}case"TensorListPushBack":{const a=i("tensorListId",t,e,s),r=i("tensor",t,e,s),n=s.getTensorList(a.id);return n.pushBack(r),[n.idTensor]}case"TensorListPopBack":{const a=i("tensorListId",t,e,s),r=i("elementShape",t,e,s),n=i("elementDType",t,e,s);return[s.getTensorList(a.id).popBack(r,n)]}case"TensorListSplit":{const a=i("tensor",t,e,s),r=i("elementShape",t,e,s),n=i("lengths",t,e,s),u=Ju(a,n,r);return s.addTensorList(u),[u.idTensor]}case"TensorListLength":{const a=i("tensorListId",t,e,s),r=s.getTensorList(a.id);return[Q(r.size(),"int32")]}case"TensorListResize":{const a=i("tensorListId",t,e,s),r=i("size",t,e,s),u=s.getTensorList(a.id).resize(r);return s.addTensorList(u),[u.idTensor]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */function Dt(t,e,s){const[a,r]=i("fusedOps",t,e,s),n=a==="biasadd",u=!n,o=r==="prelu",l=a==="fusedbatchnorm",c=i("numArgs",t,e,s);if(n){if(o&&c!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!o&&n&&c!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(l)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const p=i("strides",t,e,s),m=ke(t,e,s),h=i("dataFormat",t,e,s).toUpperCase(),d=i("dilations",t,e,s);let[b,y]=i("args",t,e,s);u&&(y=b,b=void 0);const f=i("leakyreluAlpha",t,e,s);return{stride:p,pad:m,dataFormat:h,dilations:d,biasArg:b,preluArg:y,activationFunc:r,leakyreluAlpha:f}}const Yu=(t,e,s,a=P)=>{switch(t.op){case"Conv1D":{const r=i("stride",t,e,s),n=i("pad",t,e,s),u=i("dataFormat",t,e,s).toUpperCase(),o=i("dilation",t,e,s);return[a.conv1d(i("x",t,e,s),i("filter",t,e,s),r,n,u,o)]}case"Conv2D":{const r=i("strides",t,e,s),n=ke(t,e,s),u=i("dataFormat",t,e,s).toUpperCase(),o=i("dilations",t,e,s);return[a.conv2d(i("x",t,e,s),i("filter",t,e,s),[r[1],r[2]],n,u,[o[1],o[2]])]}case"_FusedConv2D":{const{stride:r,pad:n,dataFormat:u,dilations:o,biasArg:l,preluArg:c,activationFunc:p,leakyreluAlpha:m}=Dt(t,e,s);return[a.fused.conv2d({x:i("x",t,e,s),filter:i("filter",t,e,s),strides:[r[1],r[2]],pad:n,dataFormat:u,dilations:[o[1],o[2]],bias:l,activation:p,preluActivationWeights:c,leakyreluAlpha:m})]}case"FusedDepthwiseConv2dNative":{const{stride:r,pad:n,dataFormat:u,dilations:o,biasArg:l,preluArg:c,activationFunc:p,leakyreluAlpha:m}=Dt(t,e,s);return[a.fused.depthwiseConv2d({x:i("x",t,e,s),filter:i("filter",t,e,s),strides:[r[1],r[2]],pad:n,dataFormat:u,dilations:[o[1],o[2]],bias:l,activation:p,preluActivationWeights:c,leakyreluAlpha:m})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const r=i("outputShape",t,e,s),n=i("strides",t,e,s),u=ke(t,e,s);return[a.conv2dTranspose(i("x",t,e,s),i("filter",t,e,s),r,[n[1],n[2]],u)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const r=i("strides",t,e,s),n=ke(t,e,s),u=i("dilations",t,e,s),o=i("dataFormat",t,e,s).toUpperCase();return[a.depthwiseConv2d(i("input",t,e,s),i("filter",t,e,s),[r[1],r[2]],n,o,[u[1],u[2]])]}case"Conv3D":{const r=i("strides",t,e,s),n=i("pad",t,e,s),u=i("dataFormat",t,e,s).toUpperCase(),o=i("dilations",t,e,s);return[a.conv3d(i("x",t,e,s),i("filter",t,e,s),[r[1],r[2],r[3]],n,u,[o[1],o[2],o[3]])]}case"AvgPool":{const r=i("strides",t,e,s),n=i("pad",t,e,s),u=i("kernelSize",t,e,s);return[a.avgPool(i("x",t,e,s),[u[1],u[2]],[r[1],r[2]],n)]}case"MaxPool":{const r=i("strides",t,e,s),n=i("pad",t,e,s),u=i("kernelSize",t,e,s);return[a.maxPool(i("x",t,e,s),[u[1],u[2]],[r[1],r[2]],n)]}case"MaxPoolWithArgmax":{const r=i("strides",t,e,s),n=i("pad",t,e,s),u=i("kernelSize",t,e,s),o=i("includeBatchInIndex",t,e,s),{result:l,indexes:c}=a.maxPoolWithArgmax(i("x",t,e,s),[u[1],u[2]],[r[1],r[2]],n,o);return[l,c]}case"AvgPool3D":{const r=i("strides",t,e,s),n=i("pad",t,e,s),u=i("kernelSize",t,e,s);return[a.avgPool3d(i("x",t,e,s),[u[1],u[2],u[3]],[r[1],r[2],r[3]],n)]}case"MaxPool3D":{const r=i("strides",t,e,s),n=i("pad",t,e,s),u=i("kernelSize",t,e,s);return[a.maxPool3d(i("x",t,e,s),[u[1],u[2],u[3]],[r[1],r[2],r[3]],n)]}case"Dilation2D":{const r=i("strides",t,e,s),n=i("pad",t,e,s),u=i("dilations",t,e,s),o=r[1],l=r[2],c=u[1],p=u[2];return[a.dilation2d(i("x",t,e,s),i("filter",t,e,s),[o,l],n,[c,p],"NHWC")]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const Zu=(t,e,s,a=P)=>{switch(t.op){case"Fill":{const r=i("shape",t,e,s),n=i("dtype",t,e,s),u=i("value",t,e,s);return[a.fill(r,u,n)]}case"LinSpace":{const r=i("start",t,e,s),n=i("stop",t,e,s),u=i("num",t,e,s);return[a.linspace(r,n,u)]}case"Multinomial":{const r=i("logits",t,e,s),n=i("numSamples",t,e,s),u=i("seed",t,e,s);return[a.multinomial(r,n,u)]}case"OneHot":{const r=i("indices",t,e,s),n=i("depth",t,e,s),u=i("onValue",t,e,s),o=i("offValue",t,e,s),l=i("dtype",t,e,s);return[a.oneHot(r,n,u,o,l)]}case"Ones":return[a.ones(i("shape",t,e,s),i("dtype",t,e,s))];case"OnesLike":return[a.onesLike(i("x",t,e,s))];case"RandomStandardNormal":return[a.randomStandardNormal(i("shape",t,e,s),i("dtype",t,e,s),i("seed",t,e,s))];case"RandomUniform":return[a.randomUniform(i("shape",t,e,s),i("minval",t,e,s),i("maxval",t,e,s),i("dtype",t,e,s))];case"RandomUniformInt":return[a.randomUniformInt(i("shape",t,e,s),i("minval",t,e,s),i("maxval",t,e,s),i("seed",t,e,s))];case"Range":{const r=i("start",t,e,s),n=i("stop",t,e,s),u=i("step",t,e,s);return[a.range(r,n,u,i("dtype",t,e,s))]}case"TruncatedNormal":{const r=i("shape",t,e,s),n=i("mean",t,e,s),u=i("stdDev",t,e,s),o=i("seed",t,e,s);return[a.truncatedNormal(r,n,u,i("dtype",t,e,s),o)]}case"Zeros":return[a.zeros(i("shape",t,e,s),i("dtype",t,e,s))];case"ZerosLike":return[a.zerosLike(i("x",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */function Fe(t,e,s){const a=i("boxes",t,e,s),r=i("scores",t,e,s),n=i("maxOutputSize",t,e,s),u=i("iouThreshold",t,e,s),o=i("scoreThreshold",t,e,s),l=i("softNmsSigma",t,e,s);return{boxes:a,scores:r,maxOutputSize:n,iouThreshold:u,scoreThreshold:o,softNmsSigma:l}}const Mu=async(t,e,s,a,r=P)=>{switch(t.op){case"NonMaxSuppressionV5":{const{boxes:n,scores:u,maxOutputSize:o,iouThreshold:l,scoreThreshold:c,softNmsSigma:p}=Fe(t,e,s),m=await r.image.nonMaxSuppressionWithScoreAsync(n,u,o,l,c,p);return[m.selectedIndices,m.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:n,scores:u,maxOutputSize:o,iouThreshold:l,scoreThreshold:c}=Fe(t,e,s),p=i("padToMaxOutputSize",t,e,s),m=await r.image.nonMaxSuppressionPaddedAsync(n,u,o,l,c,p);return[m.selectedIndices,m.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:n,scores:u,maxOutputSize:o,iouThreshold:l,scoreThreshold:c}=Fe(t,e,s);return[await r.image.nonMaxSuppressionAsync(n,u,o,l,c)]}case"Where":{const n=r.cast(i("condition",t,e,s),"bool"),u=[await r.whereAsync(n)];return n.dispose(),u}case"ListDiff":return r.setdiff1dAsync(i("x",t,e,s),i("y",t,e,s));default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const el=(t,e,s,a=P)=>{switch(t.op){case"LowerBound":{const r=i("sortedSequence",t,e,s),n=i("values",t,e,s);return[a.lowerBound(r,n)]}case"TopKV2":{const r=i("x",t,e,s),n=i("k",t,e,s),u=i("sorted",t,e,s),o=a.topk(r,n,u);return[o.values,o.indices]}case"UpperBound":{const r=i("sortedSequence",t,e,s),n=i("values",t,e,s);return[a.upperBound(r,n)]}case"Unique":{const r=i("x",t,e,s),n=a.unique(r);return[n.values,n.indices]}case"UniqueV2":{const r=i("x",t,e,s),n=i("axis",t,e,s),u=a.unique(r,n);return[u.values,u.indices]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const tl=(t,e,s,a=P)=>{switch(t.op){case"Const":return e[t.name];case"PlaceholderWithDefault":const r=i("default",t,e,s);return[D(t.name,e,s)||r];case"Placeholder":return[D(t.name,e,s)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const p=i("x",t,e,s);return[W(p)]}case"IdentityN":return i("x",t,e,s).map(p=>W(p));case"Snapshot":const n=i("x",t,e,s);return[W(n)];case"Shape":return[a.tensor1d(i("x",t,e,s).shape,"int32")];case"ShapeN":return i("x",t,e,s).map(p=>a.tensor1d(p.shape));case"Size":return[a.scalar(i("x",t,e,s).size,"int32")];case"Rank":return[a.scalar(i("x",t,e,s).rank,"int32")];case"NoOp":return[a.scalar(1)];case"Print":const u=i("x",t,e,s),o=i("data",t,e,s),l=i("message",t,e,s),c=i("summarize",t,e,s);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(l);for(let p=0;p<o.length;p++)console.log(Array.prototype.slice.call(o[p].dataSync()).slice(0,c));return[u];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */class sl{get id(){return this.handle.id}constructor(e,s){this.keyDType=e,this.valueDType=s,this.handle=Q(0),this.tensorMap=new Map,G(this.handle)}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return Q(this.size(),"int32")}async import(e,s){this.checkKeyAndValueTensor(e,s);const a=await e.data();return this.tensorMap.forEach(r=>r.dispose()),this.tensorMap.clear(),U(()=>{const r=ge(s),n=a.length,u=r.length;v(n===u,()=>`The number of elements doesn't match, keys has ${n} elements, the values has ${u} elements.`);for(let o=0;o<n;o++){const l=a[o],c=r[o];G(c),this.tensorMap.set(l,c)}return this.handle})}async find(e,s){this.checkKeyAndValueTensor(e,s);const a=await e.data();return U(()=>{const r=[];for(let n=0;n<a.length;n++){const u=a[n],o=this.findWithDefault(u,s);r.push(o)}return me(r)})}findWithDefault(e,s){const a=this.tensorMap.get(e);return a??s}checkKeyAndValueTensor(e,s){if(e.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(s.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${s.dtype}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 */const al=async(t,e,s,a)=>{switch(t.op){case"HashTable":case"HashTableV2":{const r=a.getHashTableHandleByName(t.name);if(r!=null)return[r];{const n=i("keyDType",t,e,s),u=i("valueDType",t,e,s),o=new sl(n,u);return a.addHashTable(t.name,o),[o.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const r=i("tableHandle",t,e,s,a),n=i("keys",t,e,s),u=i("values",t,e,s);return[await a.getHashTableById(r.id).import(n,u)]}case"LookupTableFind":case"LookupTableFindV2":{const r=i("tableHandle",t,e,s,a),n=i("keys",t,e,s),u=i("defaultValue",t,e,s);return[await a.getHashTableById(r.id).find(n,u)]}case"LookupTableSize":case"LookupTableSizeV2":{const r=i("tableHandle",t,e,s,a);return[a.getHashTableById(r.id).tensorSize()]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const rl=(t,e,s,a=P)=>{switch(t.op){case"ResizeBilinear":{const r=i("images",t,e,s),n=i("size",t,e,s),u=i("alignCorners",t,e,s),o=i("halfPixelCenters",t,e,s);return[a.image.resizeBilinear(r,[n[0],n[1]],u,o)]}case"ResizeNearestNeighbor":{const r=i("images",t,e,s),n=i("size",t,e,s),u=i("alignCorners",t,e,s),o=i("halfPixelCenters",t,e,s);return[a.image.resizeNearestNeighbor(r,[n[0],n[1]],u,o)]}case"CropAndResize":{const r=i("image",t,e,s),n=i("boxes",t,e,s),u=i("boxInd",t,e,s),o=i("cropSize",t,e,s),l=i("method",t,e,s),c=i("extrapolationValue",t,e,s);return[a.image.cropAndResize(r,n,u,o,l,c)]}case"ImageProjectiveTransformV3":{const r=i("images",t,e,s),n=i("transforms",t,e,s),u=i("outputShape",t,e,s),o=i("fillValue",t,e,s),l=i("interpolation",t,e,s),c=i("fillMode",t,e,s);return[a.image.transform(r,n,l.toLowerCase(),c.toLowerCase(),o,u)]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const nl=(t,e,s,a=P)=>{switch(t.op){case"Equal":return[a.equal(i("a",t,e,s),i("b",t,e,s))];case"NotEqual":return[a.notEqual(i("a",t,e,s),i("b",t,e,s))];case"Greater":return[a.greater(i("a",t,e,s),i("b",t,e,s))];case"GreaterEqual":return[a.greaterEqual(i("a",t,e,s),i("b",t,e,s))];case"Less":return[a.less(i("a",t,e,s),i("b",t,e,s))];case"LessEqual":return[a.lessEqual(i("a",t,e,s),i("b",t,e,s))];case"LogicalAnd":return[a.logicalAnd(i("a",t,e,s),i("b",t,e,s))];case"LogicalNot":return[a.logicalNot(i("a",t,e,s))];case"LogicalOr":return[a.logicalOr(i("a",t,e,s),i("b",t,e,s))];case"Select":case"SelectV2":return[a.where(i("condition",t,e,s),i("a",t,e,s),i("b",t,e,s))];case"BitwiseAnd":return[a.bitwiseAnd(i("a",t,e,s),i("b",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const il=(t,e,s,a=P)=>{switch(t.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[a.matMul(i("a",t,e,s),i("b",t,e,s),i("transposeA",t,e,s),i("transposeB",t,e,s))];case"Einsum":return[a.einsum(i("equation",t,e,s),...i("tensors",t,e,s))];case"Transpose":return[a.transpose(i("x",t,e,s),i("perm",t,e,s))];case"_FusedMatMul":const[r,n]=i("fusedOps",t,e,s),u=r==="biasadd",o=n==="prelu",l=i("numArgs",t,e,s),c=i("leakyreluAlpha",t,e,s);if(u){if(o&&l!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!o&&l!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[p,m]=i("args",t,e,s);return[a.fused.matMul({a:i("a",t,e,s),b:i("b",t,e,s),transposeA:i("transposeA",t,e,s),transposeB:i("transposeB",t,e,s),bias:p,activation:n,preluActivationWeights:m,leakyreluAlpha:c})];case"MatrixBandPart":return[a.linalg.bandPart(i("a",t,e,s),i("numLower",t,e,s),i("numUpper",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const ol=(t,e,s,a=P)=>{switch(t.op){case"EuclideanNorm":return[a.euclideanNorm(i("x",t,e,s),i("axis",t,e,s),i("keepDims",t,e,s))];case"FusedBatchNorm":case"FusedBatchNormV2":return[a.batchNorm(i("x",t,e,s),i("mean",t,e,s),i("variance",t,e,s),i("offset",t,e,s),i("scale",t,e,s),i("epsilon",t,e,s))];case"FusedBatchNormV3":return[a.batchNorm(i("x",t,e,s),i("mean",t,e,s),i("variance",t,e,s),i("offset",t,e,s),i("scale",t,e,s),i("epsilon",t,e,s))];case"LRN":return[a.localResponseNormalization(i("x",t,e,s),i("radius",t,e,s),i("bias",t,e,s),i("alpha",t,e,s),i("beta",t,e,s))];case"Softmax":return[a.softmax(i("x",t,e,s))];case"LogSoftmax":return[a.logSoftmax(i("x",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
 */const ul=(t,e,s,a=P)=>{switch(t.op){case"RaggedGather":{const{outputNestedSplits:r,outputDenseValues:n}=a.raggedGather(i("paramsNestedSplits",t,e,s),i("paramsDenseValues",t,e,s),i("indices",t,e,s),i("outputRaggedRank",t,e,s));return r.concat(n)}case"RaggedRange":{const{rtNestedSplits:r,rtDenseValues:n}=a.raggedRange(i("starts",t,e,s),i("limits",t,e,s),i("splits",t,e,s));return[r,n]}case"RaggedTensorToTensor":return[a.raggedTensorToTensor(i("shape",t,e,s),i("values",t,e,s),i("defaultValue",t,e,s),i("rowPartitionTensors",t,e,s),i("rowPartitionTypes",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const ll=(t,e,s,a=P)=>{switch(t.op){case"Max":{const o=i("axis",t,e,s),l=i("keepDims",t,e,s);return[a.max(i("x",t,e,s),o,l)]}case"Mean":{const o=i("axis",t,e,s),l=i("keepDims",t,e,s);return[a.mean(i("x",t,e,s),o,l)]}case"Min":{const o=i("axis",t,e,s),l=i("keepDims",t,e,s);return[a.min(i("x",t,e,s),o,l)]}case"Sum":{const o=i("axis",t,e,s),l=i("keepDims",t,e,s);return[a.sum(i("x",t,e,s),o,l)]}case"All":{const o=i("axis",t,e,s),l=i("keepDims",t,e,s);return[a.all(i("x",t,e,s),o,l)]}case"Any":{const o=i("axis",t,e,s),l=i("keepDims",t,e,s);return[a.any(i("x",t,e,s),o,l)]}case"ArgMax":{const o=i("axis",t,e,s);return[a.argMax(i("x",t,e,s),o)]}case"ArgMin":{const o=i("axis",t,e,s);return[a.argMin(i("x",t,e,s),o)]}case"Prod":{const o=i("axis",t,e,s),l=i("keepDims",t,e,s);return[a.prod(i("x",t,e,s),o,l)]}case"Cumprod":{const o=i("axis",t,e,s),l=i("exclusive",t,e,s),c=i("reverse",t,e,s);return[a.cumprod(i("x",t,e,s),o,l,c)]}case"Cumsum":{const o=i("axis",t,e,s),l=i("exclusive",t,e,s),c=i("reverse",t,e,s);return[a.cumsum(i("x",t,e,s),o,l,c)]}case"Bincount":const r=i("x",t,e,s),n=i("weights",t,e,s),u=i("size",t,e,s);return[a.bincount(r,n,u)];case"DenseBincount":{const o=i("x",t,e,s),l=i("weights",t,e,s),c=i("size",t,e,s),p=i("binaryOutput",t,e,s);return[a.denseBincount(o,l,c,p)]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const cl=(t,e,s,a=P)=>{switch(t.op){case"ConcatV2":case"Concat":{const r=i("n",t,e,s),n=i("axis",t,e,s);let u=i("tensors",t,e,s);return u=u.slice(0,r),[a.concat(u,n)]}case"Gather":{const r=i("x",t,e,s),n=i("indices",t,e,s);return[a.gather(r,a.cast(n,"int32"),0)]}case"GatherV2":{const r=i("axis",t,e,s),n=i("batchDims",t,e,s),u=i("x",t,e,s),o=i("indices",t,e,s);return[a.gather(u,a.cast(o,"int32"),r,n)]}case"Reverse":{const r=i("dims",t,e,s),n=[];for(let o=0;o<r.length;o++)r[o]&&n.push(o);const u=i("x",t,e,s);return[a.reverse(u,n)]}case"ReverseV2":{const r=i("axis",t,e,s),n=i("x",t,e,s);return[a.reverse(n,r)]}case"Slice":{const r=i("begin",t,e,s),n=i("size",t,e,s);return[a.slice(i("x",t,e,s),r,n)]}case"StridedSlice":{const r=i("begin",t,e,s),n=i("end",t,e,s),u=i("strides",t,e,s),o=i("beginMask",t,e,s),l=i("endMask",t,e,s),c=i("ellipsisMask",t,e,s),p=i("newAxisMask",t,e,s),m=i("shrinkAxisMask",t,e,s),h=i("x",t,e,s);return[a.stridedSlice(h,r,n,u,o,l,c,p,m)]}case"Pack":return U(()=>{const r=i("axis",t,e,s),n=i("tensors",t,e,s),u=n[0].shape,o=a.squeeze(n[0]).shape,l=n.map(c=>{const p=le(c.shape,u);if(!p&&!le(a.squeeze(c).shape,o))throw new Error("the input tensors shape does not match");return p?c:a.reshape(c,u)});return[a.stack(l,r)]});case"Unpack":{const r=i("axis",t,e,s),n=i("tensor",t,e,s);return a.unstack(n,r)}case"Tile":{const r=i("reps",t,e,s);return[a.tile(i("x",t,e,s),r)]}case"Split":case"SplitV":{const r=i("axis",t,e,s),n=i("numOrSizeSplits",t,e,s),u=i("x",t,e,s);return a.split(u,n,r)}case"ScatterNd":{const r=i("indices",t,e,s),n=i("values",t,e,s),u=i("shape",t,e,s);return[a.scatterND(r,n,u)]}case"GatherNd":{const r=i("x",t,e,s),n=i("indices",t,e,s);return[a.gatherND(r,n)]}case"SparseToDense":{const r=i("sparseIndices",t,e,s),n=i("outputShape",t,e,s),u=i("sparseValues",t,e,s),o=i("defaultValue",t,e,s);return[a.sparseToDense(r,u,n,u.dtype===o.dtype?o:a.cast(o,u.dtype))]}case"TensorScatterUpdate":{const r=i("indices",t,e,s),n=i("values",t,e,s),u=i("tensor",t,e,s);return[a.tensorScatterUpdate(u,r,n)]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
 */const pl=(t,e,s,a=P)=>{switch(t.op){case"SparseFillEmptyRows":{const{outputIndices:r,outputValues:n,emptyRowIndicator:u,reverseIndexMap:o}=a.sparse.sparseFillEmptyRows(i("indices",t,e,s),i("values",t,e,s),i("denseShape",t,e,s),i("defaultValue",t,e,s));return[r,n,u,o]}case"SparseReshape":{const{outputIndices:r,outputShape:n}=a.sparse.sparseReshape(i("inputIndices",t,e,s),i("inputShape",t,e,s),i("newShape",t,e,s));return[r,n]}case"SparseSegmentMean":return[a.sparse.sparseSegmentMean(i("data",t,e,s),i("indices",t,e,s),i("segmentIds",t,e,s))];case"SparseSegmentSum":return[a.sparse.sparseSegmentSum(i("data",t,e,s),i("indices",t,e,s),i("segmentIds",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const ml=(t,e,s,a=P)=>{switch(t.op){case"FFT":return[a.fft(i("x",t,e,s))];case"IFFT":return[a.ifft(i("x",t,e,s))];case"RFFT":return[a.rfft(i("x",t,e,s))];case"IRFFT":return[a.irfft(i("x",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
 */const hl=(t,e,s,a=P)=>{switch(t.op){case"StaticRegexReplace":return[a.string.staticRegexReplace(i("input",t,e,s),i("pattern",t,e,s),i("rewrite",t,e,s),i("replaceGlobal",t,e,s))];case"StringNGrams":{const{nGrams:r,nGramsSplits:n}=a.string.stringNGrams(i("data",t,e,s),i("dataSplits",t,e,s),i("separator",t,e,s),i("nGramWidths",t,e,s),i("leftPad",t,e,s),i("rightPad",t,e,s),i("padWidth",t,e,s),i("preserveShortSequences",t,e,s));return[r,n]}case"StringSplit":{const{indices:r,values:n,shape:u}=a.string.stringSplit(i("input",t,e,s),i("delimiter",t,e,s),i("skipEmpty",t,e,s));return[r,n,u]}case"StringToHashBucketFast":return[a.string.stringToHashBucketFast(i("input",t,e,s),i("numBuckets",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */const dl=(t,e,s,a=P)=>{switch(t.op){case"Cast":return[a.cast(i("x",t,e,s),i("dtype",t,e,s))];case"ExpandDims":{const r=i("axis",t,e,s);return[a.expandDims(i("x",t,e,s),r)]}case"Squeeze":{const r=i("axis",t,e,s);return[a.squeeze(i("x",t,e,s),r)]}case"Reshape":return[a.reshape(i("x",t,e,s),i("shape",t,e,s))];case"EnsureShape":return[a.ensureShape(i("x",t,e,s),i("shape",t,e,s))];case"MirrorPad":return[a.mirrorPad(i("x",t,e,s),i("padding",t,e,s),i("mode",t,e,s))];case"PadV2":case"Pad":return[a.pad(i("x",t,e,s),i("padding",t,e,s),i("constantValue",t,e,s))];case"SpaceToBatchND":{const r=i("blockShape",t,e,s),n=i("paddings",t,e,s);return[a.spaceToBatchND(i("x",t,e,s),r,n)]}case"BatchToSpaceND":{const r=i("blockShape",t,e,s),n=i("crops",t,e,s);return[a.batchToSpaceND(i("x",t,e,s),r,n)]}case"DepthToSpace":{const r=i("blockSize",t,e,s),n=i("dataFormat",t,e,s).toUpperCase();return[a.depthToSpace(i("x",t,e,s),r,n)]}case"BroadcastTo":return[a.broadcastTo(i("x",t,e,s),i("shape",t,e,s))];case"BroadcastArgs":return[a.broadcastArgs(i("s0",t,e,s),i("s1",t,e,s))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
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
 */function Pt(t,e,s,a,r=U){const n=((u,o,l)=>{switch(u.category){case"arithmetic":return r(()=>Uu(u,o,l));case"basic_math":return r(()=>Hu(u,o,l));case"control":return Xu(u,o,l);case"convolution":return r(()=>Yu(u,o,l));case"creation":return r(()=>Zu(u,o,l));case"dynamic":return Mu(u,o,l);case"evaluation":return r(()=>el(u,o,l));case"image":return r(()=>rl(u,o,l));case"graph":return r(()=>tl(u,o,l));case"logical":return r(()=>nl(u,o,l));case"matrices":return r(()=>il(u,o,l));case"normalization":return r(()=>ol(u,o,l));case"ragged":return r(()=>ul(u,o,l));case"reduction":return r(()=>ll(u,o,l));case"slice_join":return r(()=>cl(u,o,l));case"sparse":return r(()=>pl(u,o,l));case"spectral":return r(()=>ml(u,o,l));case"string":return r(()=>hl(u,o,l));case"transformation":return r(()=>dl(u,o,l));case"hash_table":return al(u,o,l,a);case"custom":const c=ys(u.op);if(c&&c.customExecutor)return c.customExecutor(new qu(u,o,l));throw TypeError(`Custom op ${u.op} is not registered.`);default:throw TypeError(`Unknown op '${u.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(t,e,s);return Ee(n)?n.then(u=>[].concat(u)):[].concat(n)}class jt{constructor(e={},s={},a={},r={},n){this.weightMap=e,this.tensorArrayMap=s,this.tensorListMap=a,this.functionMap=r,this.parseNodeNameCache=n,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,s){return{id:e,frameName:s,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const e=[];for(let s=0;s<this.contexts.length-1;s++){const a=this.contexts.slice(0,this.contexts.length-s);e.push(this.contextIdforContexts(a))}e.push(""),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(s=>s.id===0&&s.iterationId===0?"":`${s.frameName}-${s.iterationId}`).join("/"):""}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(const s in this.tensorArrayMap)this.tensorArrayMap[s].clearAndClose(e);for(const s in this.tensorListMap)this.tensorListMap[s].clearAndClose(e)}}/**
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
 */function xt(t,e,s,a){const r=new Set,n=[];let u=null,o=null;const l=new Set,c=new Set(Object.keys(t).map(h=>z(h)[0]));a=a||[];const p=new Set(a.map(h=>z(h.name)[0])),m=[...e];for(;m.length>0;){const h=m.pop();if((Y(h)||Sl(h)||vl(h))&&u==null&&(u=h,o=u.children.map(d=>d.name).filter(d=>r.has(d))),r.add(h.name),s[h.name]==null&&!c.has(h.name)&&!p.has(h.name)){if(h.inputs.length===0){n.push(h.name);continue}h.inputs.forEach(d=>{l.has(d.name)||(l.add(d.name),m.push(d))})}}return{inputs:t,outputs:e,usedNodes:r,missingInputs:n,dynamicNode:u,syncInputs:o}}function fl(t,e){const{usedNodes:s,inputs:a}=e,r=Object.keys(a).map(f=>z(f)[0]).map(f=>t.nodes[f]),n=t.initNodes||[],u=f=>s.has(typeof f=="string"?f:f.name);function o(f){return[...new Map(f.map(N=>[N.name,N])).values()]}const l=o([...r,...t.weights,...n]).filter(u),c=o([...l,...Object.values(t.nodes)]).filter(u),p=new Map(c.map(f=>[f.name,f])),m={};for(const f of c){m[f.name]=m[f.name]||0;for(const N of f.children)u(N)||(m[N.name]=Number.POSITIVE_INFINITY),m[N.name]=(m[N.name]||0)+1}const h=Object.entries(m).filter(([,f])=>f===0).map(([f])=>f),d=[...h];for(;h.length>0;){const f=h.pop(),N=p.get(f);for(const S of N.children.filter(u))--m[S.name]===0&&(d.push(S.name),h.push(S.name))}const b=d.map(f=>p.get(f)),y=yl(b,l);return gl(y,l),y}function yl(t,e){const s=new Map(t.map(u=>[u.name,u])),a=e.map(u=>u.name),r=new Set(a);for(;a.length>0;){const u=a.pop(),o=s.get(u);for(const l of o.children)!s.has(l.name)||r.has(l.name)||(r.add(l.name),a.push(l.name))}return t.filter(u=>r.has(u.name))}class we extends Error{constructor(e){super(`NodesExecutionOrderError: ${e}`)}}function gl(t,e){const s=new Map(t.map((o,l)=>[o.name,l])),a=new Set(e.map(o=>o.name)),r=o=>a.has(typeof o=="string"?o:o.name),n=new Set(t.map(o=>o.name)),u=o=>n.has(typeof o=="string"?o:o.name);for(const o of t){for(const l of o.children.filter(u)){if(!s.has(l.name))throw new we(`Child ${l.name} of node ${o.name} is unreachable.`);if(s.get(o.name)>s.get(l.name))throw new we(`Node ${o.name} is scheduled to run after its child ${l.name}.`)}if(!r(o))for(const l of o.inputs){if(!s.has(l.name))throw new we(`Input ${l.name} of node ${o.name} is unreachable.`);if(s.get(l.name)>s.get(o.name))throw new we(`Node ${o.name} is scheduled to run before its input ${l.name}.`)}}}function bl(t){const e=new Map(t.map((o,l)=>[o.name,l])),s=Number.MAX_SAFE_INTEGER,a=t.map((o,l)=>Y(o)?s:l),r=o=>{const l=a[e.get(o.name)];return l??-1},n=t.map((o,l)=>o.children.map(r).reduce((c,p)=>Math.max(c,p),a[l])),u=new Map;for(let o=0;o<t.length;++o){const l=n[o];if(l===s)continue;const c=t[o],p=t[l];u.has(p.name)||u.set(p.name,[]),u.get(p.name).push(c)}return u}const Nl=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),wl=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),Tl=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function Y(t){return Nl.has(t.op)}function Sl(t){return wl.has(t.op)}function vl(t){return Tl.has(t.op)}/**
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
 */class Ie{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){const s=Object.keys(e).map(a=>e[a].map(r=>r.id));this._weightIds=[].concat(...s),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{const s=e.signatureKey||e.name;return e.defaultOutput?`${s}:${e.defaultOutput}`:s})}get functions(){return Object.keys(this._functions).reduce((e,s)=>(e[s]=this._functions[s].signature,e),{})}constructor(e,s){this.graph=e,this.parent=s,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=e.outputs,this._inputs=e.inputs,this._initNodes=e.initNodes,this._signature=e.signature,this._functions=e.functions,e.functions!=null&&Object.keys(e.functions).forEach(a=>{this._functionExecutorMap[a]=new Ie(e.functions[a],this)})}getCompilationKey(e,s){const a=e.map(n=>n.name).sort(),r=s.map(n=>n.name).sort();return a.join(this.SEPARATOR)+"--"+r.join(this.SEPARATOR)}compile(e,s){const a=xt(e,s,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:n,syncInputs:u}=a;if(n!=null)throw new Error(`This execution contains the node '${n.name}', which has the dynamic op '${n.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${u}]`);if(r.length>0){const c=s.map(m=>m.name),p=Object.keys(e);throw new Error(`Cannot compute the outputs [${c}] from the provided inputs [${p}]. Missing the following inputs: [${r}]`)}const o=fl(this.graph,a),l=bl(o);return{orderedNodes:o,nodeLiveUntilMap:l}}cloneAndKeepTensor(e){if(e==null)return null;const s=e.clone();return G(s),s}cloneTensorList(e){return e?e.map(a=>this.cloneAndKeepTensor(a)):null}cloneTensorMap(e){return Object.fromEntries(Object.entries(e).map(([s,a])=>[s,this.cloneTensorList(a)]))}execute(e,s){this.disposeIntermediateTensors(),e=this.mapInputs(e);const a=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),s=this.mapOutputs(s),this.checkOutputs(s);const r=a.map(h=>this.graph.nodes[z(h)[0]]),n=s.map(h=>z(h)[0]),u=new Set(n);let o=n.map(h=>this.graph.nodes[h]);o.length===0&&(o=this._outputs);const l=this.getCompilationKey(r,o);let c=this.compiledMap.get(l);c==null&&(c=this.compile(e,o),this.compiledMap.set(l,c));try{this.keepIntermediateTensors=X().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(h){this.keepIntermediateTensors=!1,console.warn(h.message)}const p={},m={};return U(()=>{const h=new jt(this.weightMap,p,m,this.functionExecutorMap,this.parseNodeNameCache),d=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(e).forEach(N=>{const[S,k]=z(N,h),T=[];T[k]=e[N],d[S]=T,this.keepIntermediateTensors&&(this.clonedTensorsMap[S]=this.cloneTensorList(T))});const b=this.getFrozenTensorIds(d),{orderedNodes:y,nodeLiveUntilMap:f}=c;for(const N of y){if(d[N.name])continue;const S=Pt(N,d,h,this._resourceManager);if(Ee(S))throw new Error(`The execution of the op '${N.op}' returned a promise. Please use model.executeAsync() instead.`);d[N.name]=S,this.keepIntermediateTensors&&(this.clonedTensorsMap[N.name]=this.cloneTensorList(S)),this.checkTensorForDisposalWithNodeLiveUntilInfo(N,d,h,b,u,f.get(N.name))}return this.parent==null&&h.dispose(b),s.map(N=>D(N,d,h))})}getFrozenTensorIds(e){const s=[].concat.apply([],Object.keys(e).map(a=>e[a]).map(a=>a.map(r=>r.id)));return new Set(s)}checkTensorForDisposal(e,s,a,r,n,u,o){if(!(Y(s)||u.has(e))){for(const l of a[e])l!=null&&(o[l.id]=(o[l.id]||0)+s.children.length);for(const l of s.inputs){if(Y(l))continue;const c=Et(l.name,a,r);if(c!=null)for(const p of c){if(!p||p.kept||n.has(p.id))continue;const m=o[p.id];m===1?(p.dispose(),delete o[p.id]):m!=null&&o[p.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(e,s,a,r,n,u){function o(l){return Y(l)||n.has(l.name)}if(!(Y(e)||u==null))for(const l of u){if(o(l))continue;const c=Et(l.name,s,a);for(const p of c)!p||p.kept||r.has(p.id)||p.dispose()}}async executeAsync(e,s){return this._executeAsync(e,s)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(e=>{for(const s of e)s&&!s.isDisposed&&s.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(e,s,a=!1,r={},n={}){this.disposeIntermediateTensors(),a||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),s=this.mapOutputs(s),this.checkOutputs(s));try{this.keepIntermediateTensors=X().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(h){this.keepIntermediateTensors=!1,console.warn(h.message)}const u=new jt(this.weightMap,r,n,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const o=await this.executeWithControlFlow(e,u,s,a),l=s.map(h=>D(h,o,u)),c=l.map(h=>h.id),p=Object.keys(e).map(h=>e[h].id),m=new Set([...c,...p,...this.weightIds]);return Object.values(o).forEach(h=>{h.forEach(d=>{d&&!d.isDisposed&&!m.has(d.id)&&d.dispose()})}),this.parent==null&&u.dispose(m),l}async executeFunctionAsync(e,s,a){const r=e.reduce((n,u,o)=>(n[this.inputs[o].name]=u,n),{});return this._executeAsync(r,this.outputNodes,!0,s,a)}async executeWithControlFlow(e,s,a,r){const n=Object.keys(e),u=n.map(T=>this.graph.nodes[z(T)[0]]),o=a.map(T=>z(T)[0]),l=new Set(o);let c=o.map(T=>this.graph.nodes[T]);c.length===0&&(c=this._outputs);const{usedNodes:p,missingInputs:m,dynamicNode:h,syncInputs:d}=xt(e,c,this.weightMap,this._initNodes),b=[...u,...this.graph.weights,...this._initNodes||[]].map(T=>({node:T,contexts:s.currentContext})),y=Object.assign({},this.weightMap);Object.keys(e).forEach(T=>{const[_,j]=z(T),A=[];A[j]=e[T],y[_]=A});const f={},N=this.getFrozenTensorIds(y),S={};for(;b.length>0;){const T=this.processStack(u,b,s,y,S,N,l,f,p);await Promise.all(T)}h==null&&!r&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const k=c.filter(T=>!Y(T)&&!D(T.name,y,s)).map(T=>T.name);if(k.length>0){let T="";throw h!=null&&(T=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${d}]`),new Error(`Cannot compute the outputs [${k}] from the provided inputs [${n}]. Consider providing the following inputs: [${m}]. ${T}`)}return y}processStack(e,s,a,r,n,u,o,l,c){const p=[];for(;s.length>0;){const m=s.pop();a.currentContext=m.contexts;let h="";if(m.node.op==="Enter"&&i("isConstant",m.node,r,a)&&([h]=H(m.node.name,a)),r[m.node.name]==null){const d=Pt(m.node,r,a,this._resourceManager);h||([h]=H(m.node.name,a));const b=a.currentContext;Ee(d)?p.push(d.then(y=>(r[h]=y,this.keepIntermediateTensors&&(this.clonedTensorsMap[h]=this.cloneTensorList(y)),a.currentContext=b,this.checkTensorForDisposal(h,m.node,r,a,u,o,l),this.processChildNodes(m.node,s,a,r,n,c),y))):(r[h]=d,this.keepIntermediateTensors&&(this.clonedTensorsMap[h]=this.cloneTensorList(d)),this.checkTensorForDisposal(h,m.node,r,a,u,o,l),this.processChildNodes(m.node,s,a,r,n,c))}else this.processChildNodes(m.node,s,a,r,n,c)}return p}processChildNodes(e,s,a,r,n,u){e.children.forEach(o=>{const[l]=H(o.name,a);n[l]||!u.has(o.name)||(o.op==="Merge"?o.inputNames.some(c=>!!D(c,r,a))&&(n[l]=!0,s.push({contexts:a.currentContext,node:o})):o.inputNames.every(c=>!!D(c,r,a))&&(n[l]=!0,s.push({contexts:a.currentContext,node:o})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(s=>s.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(s=>{const a=e[s],[r]=z(s),n=this.graph.nodes[r];if(n.attrParams.shape&&n.attrParams.shape.value){const u=n.attrParams.shape.value,o=u.length===a.shape.length&&a.shape.every((l,c)=>u[c]===-1||u[c]===l);v(o,()=>`The shape of dict['${n.name}'] provided in model.execute(dict) must be [${u}], but was [${a.shape}]`)}n.attrParams.dtype&&n.attrParams.dtype.value&&v(a.dtype===n.attrParams.dtype.value,()=>`The dtype of dict['${n.name}'] provided in model.execute(dict) must be ${n.attrParams.dtype.value}, but was ${a.dtype}`)})}mapInputs(e){var s,a;const r={};for(const n in e){const u=(a=(s=this._signature)===null||s===void 0?void 0:s.inputs)===null||a===void 0?void 0:a[n];u!=null?r[u.name]=e[n]:r[n]=e[n]}return r}checkInputs(e){const s=Object.keys(e).filter(a=>{const[r]=z(a);return this.graph.nodes[r]==null});if(s.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${s}] that are not part of graph`)}mapOutputs(e){return e.map(s=>{var a,r;const n=(r=(a=this._signature)===null||a===void 0?void 0:a.outputs)===null||r===void 0?void 0:r[s];return n!=null?n.name:s},{})}checkOutputs(e){e.forEach(s=>{const[a]=z(s);if(!this.graph.nodes[a])throw new Error(`The output '${s}' is not found in the graph`)})}}class Ol{constructor(e={},s={}){this.hashTableNameToHandle=e,this.hashTableMap=s}addHashTable(e,s){this.hashTableNameToHandle[e]=s.handle,this.hashTableMap[s.id]=s}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(const e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(const e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}}/**
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
 */const kl="?tfjs-format=file",_l="model.json";class El{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(e,s={},a=fs){this.modelUrl=e,this.loadOptions=s,this.version="n/a",this.io=a,s==null&&(this.loadOptions={}),this.resourceManager=new Ol}findIOHandler(){const e=this.modelUrl;if(e.load!=null)this.handler=e;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{const s=this.io.getLoadHandlers(e,this.loadOptions);if(s.length===0)s.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(s.length>1)throw new Error(`Found more than one (${s.length}) load handlers for URL '${[e]}'`);this.handler=s[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const e=this.handler.load();return Ee(e)?e.then(s=>s.getWeightStream==null?this.loadSync(s):this.loadStreaming(s)):this.loadSync(e)}loadSync(e){const s=this.io.decodeWeights(e.weightData,e.weightSpecs);return this.loadWithWeightMap(e,s)}async loadStreaming(e){if(e.getWeightStream==null)throw new Error("Model artifacts missing streamWeights function");const s=await ns(e.getWeightStream(),e.weightSpecs);return this.loadWithWeightMap(e,s)}loadWithWeightMap(e,s){this.artifacts=e;const a=this.artifacts.modelTopology;let r=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const n=this.artifacts.userDefinedMetadata;n.signature!=null&&(r=n.signature),n.structuredOutputKeys!=null&&(this.structuredOutputKeys=n.structuredOutputKeys)}if(this.signature=r,this.version=`${a.versions.producer}.${a.versions.minConsumer}`,this.executor=new Ie(At.Instance.transformGraph(a,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(s),this.executor.resourceManager=this.resourceManager,e.modelInitializer!=null&&e.modelInitializer.node!=null){const n=At.Instance.transformGraph(e.modelInitializer);this.initializer=new Ie(n),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=e.initializerSignature}return!0}async save(e,s){if(typeof e=="string"){const a=this.io.getSaveHandlers(e);if(a.length===0)throw new Error(`Cannot find any save handlers for URL '${e}'`);if(a.length>1)throw new Error(`Found more than one (${a.length}) save handlers for URL '${e}'`);e=a[0]}if(e.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}addStructuredOutputNames(e){if(this.structuredOutputKeys){const s=e instanceof pe?[e]:e,a={};return s.forEach((r,n)=>a[this.structuredOutputKeys[n]]=r),a}return e}predict(e,s){const a=this.execute(e,this.outputNodes);return this.addStructuredOutputNames(a)}async predictAsync(e,s){const a=await this.executeAsync(e,this.outputNodes);return this.addStructuredOutputNames(a)}normalizeInputs(e){var s;if(!(e instanceof pe)&&!Array.isArray(e)){const n=(s=this.signature)===null||s===void 0?void 0:s.inputs;if(n!=null)for(const u in n){const o=n[u];o.resourceId!=null&&(e[u]=this.resourceIdToCapturedInput[o.resourceId])}return e}e=Array.isArray(e)?e:[e];const a=Object.keys(this.resourceIdToCapturedInput).length;if(e.length+a!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-a} non-resource placeholders, while there are ${e.length} input tensors provided.`);let r=0;return this.inputNodes.reduce((n,u)=>{var o,l,c;const p=(c=(l=(o=this.signature)===null||o===void 0?void 0:o.inputs)===null||l===void 0?void 0:l[u])===null||c===void 0?void 0:c.resourceId;return p!=null?n[u]=this.resourceIdToCapturedInput[p]:n[u]=e[r++],n},{})}normalizeOutputs(e){return e=e||this.outputNodes,Array.isArray(e)?e:[e]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(e){if(this.resourceIdToCapturedInput={},this.initializerSignature){const s=this.initializerSignature.outputs,a=Object.keys(s);for(let r=0;r<a.length;r++){const n=a[r],u=s[n];this.resourceIdToCapturedInput[u.resourceId]=e[r]}}}execute(e,s){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),e=this.normalizeInputs(e),s=this.normalizeOutputs(s);const a=this.executor.execute(e,s);return a.length>1?a:a[0]}async executeAsync(e,s){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),e=this.normalizeInputs(e),s=this.normalizeOutputs(s);const a=await this.executor.executeAsync(e,s);return a.length>1?a:a[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((s,a)=>(s[a]=[e[a]],s),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&li(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function $t(t,e={},s=fs){if(t==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");e==null&&(e={}),e.fromTFHub&&typeof t=="string"&&(t=Al(t));const a=new El(t,e,s);return await a.load(),a}function Al(t){return t.endsWith("/")||(t=t+"/"),`${t}${_l}${kl}`}var be=class{constructor(){this.listeners=new Set,this.subscribe=this.subscribe.bind(this)}subscribe(t){return this.listeners.add(t),this.onSubscribe(),()=>{this.listeners.delete(t),this.onUnsubscribe()}}hasListeners(){return this.listeners.size>0}onSubscribe(){}onUnsubscribe(){}},Il={setTimeout:(t,e)=>setTimeout(t,e),clearTimeout:t=>clearTimeout(t),setInterval:(t,e)=>setInterval(t,e),clearInterval:t=>clearInterval(t)},Cl=class{#e=Il;#t=!1;setTimeoutProvider(t){this.#e=t}setTimeout(t,e){return this.#e.setTimeout(t,e)}clearTimeout(t){this.#e.clearTimeout(t)}setInterval(t,e){return this.#e.setInterval(t,e)}clearInterval(t){this.#e.clearInterval(t)}},ee=new Cl;function Dl(t){setTimeout(t,0)}var ae=typeof window>"u"||"Deno"in globalThis;function R(){}function Pl(t,e){return typeof t=="function"?t(e):t}function Ye(t){return typeof t=="number"&&t>=0&&t!==1/0}function Ns(t,e){return Math.max(t+(e||0)-Date.now(),0)}function J(t,e){return typeof t=="function"?t(e):t}function B(t,e){return typeof t=="function"?t(e):t}function Ft(t,e){const{type:s="all",exact:a,fetchStatus:r,predicate:n,queryKey:u,stale:o}=t;if(u){if(a){if(e.queryHash!==dt(u,e.options))return!1}else if(!de(e.queryKey,u))return!1}if(s!=="all"){const l=e.isActive();if(s==="active"&&!l||s==="inactive"&&l)return!1}return!(typeof o=="boolean"&&e.isStale()!==o||r&&r!==e.state.fetchStatus||n&&!n(e))}function zt(t,e){const{exact:s,status:a,predicate:r,mutationKey:n}=t;if(n){if(!e.options.mutationKey)return!1;if(s){if(he(e.options.mutationKey)!==he(n))return!1}else if(!de(e.options.mutationKey,n))return!1}return!(a&&e.state.status!==a||r&&!r(e))}function dt(t,e){return(e?.queryKeyHashFn||he)(t)}function he(t){return JSON.stringify(t,(e,s)=>Me(s)?Object.keys(s).sort().reduce((a,r)=>(a[r]=s[r],a),{}):s)}function de(t,e){return t===e?!0:typeof t!=typeof e?!1:t&&e&&typeof t=="object"&&typeof e=="object"?Object.keys(e).every(s=>de(t[s],e[s])):!1}var jl=Object.prototype.hasOwnProperty;function ws(t,e){if(t===e)return t;const s=Rt(t)&&Rt(e);if(!s&&!(Me(t)&&Me(e)))return e;const r=(s?t:Object.keys(t)).length,n=s?e:Object.keys(e),u=n.length,o=s?new Array(u):{};let l=0;for(let c=0;c<u;c++){const p=s?c:n[c],m=t[p],h=e[p];if(m===h){o[p]=m,(s?c<r:jl.call(t,p))&&l++;continue}if(m===null||h===null||typeof m!="object"||typeof h!="object"){o[p]=h;continue}const d=ws(m,h);o[p]=d,d===m&&l++}return r===u&&l===r?t:o}function Ze(t,e){if(!e||Object.keys(t).length!==Object.keys(e).length)return!1;for(const s in t)if(t[s]!==e[s])return!1;return!0}function Rt(t){return Array.isArray(t)&&t.length===Object.keys(t).length}function Me(t){if(!Lt(t))return!1;const e=t.constructor;if(e===void 0)return!0;const s=e.prototype;return!(!Lt(s)||!s.hasOwnProperty("isPrototypeOf")||Object.getPrototypeOf(t)!==Object.prototype)}function Lt(t){return Object.prototype.toString.call(t)==="[object Object]"}function xl(t){return new Promise(e=>{ee.setTimeout(e,t)})}function et(t,e,s){return typeof s.structuralSharing=="function"?s.structuralSharing(t,e):s.structuralSharing!==!1?ws(t,e):e}function $l(t,e,s=0){const a=[...t,e];return s&&a.length>s?a.slice(1):a}function Fl(t,e,s=0){const a=[e,...t];return s&&a.length>s?a.slice(0,-1):a}var ft=Symbol();function Ts(t,e){return!t.queryFn&&e?.initialPromise?()=>e.initialPromise:!t.queryFn||t.queryFn===ft?()=>Promise.reject(new Error(`Missing queryFn: '${t.queryHash}'`)):t.queryFn}function zl(t,e){return typeof t=="function"?t(...e):!!t}var Rl=class extends be{#e;#t;#s;constructor(){super(),this.#s=t=>{if(!ae&&window.addEventListener){const e=()=>t();return window.addEventListener("visibilitychange",e,!1),()=>{window.removeEventListener("visibilitychange",e)}}}}onSubscribe(){this.#t||this.setEventListener(this.#s)}onUnsubscribe(){this.hasListeners()||(this.#t?.(),this.#t=void 0)}setEventListener(t){this.#s=t,this.#t?.(),this.#t=t(e=>{typeof e=="boolean"?this.setFocused(e):this.onFocus()})}setFocused(t){this.#e!==t&&(this.#e=t,this.onFocus())}onFocus(){const t=this.isFocused();this.listeners.forEach(e=>{e(t)})}isFocused(){return typeof this.#e=="boolean"?this.#e:globalThis.document?.visibilityState!=="hidden"}},yt=new Rl;function tt(){let t,e;const s=new Promise((r,n)=>{t=r,e=n});s.status="pending",s.catch(()=>{});function a(r){Object.assign(s,r),delete s.resolve,delete s.reject}return s.resolve=r=>{a({status:"fulfilled",value:r}),t(r)},s.reject=r=>{a({status:"rejected",reason:r}),e(r)},s}var Ll=Dl;function Vl(){let t=[],e=0,s=o=>{o()},a=o=>{o()},r=Ll;const n=o=>{e?t.push(o):r(()=>{s(o)})},u=()=>{const o=t;t=[],o.length&&r(()=>{a(()=>{o.forEach(l=>{s(l)})})})};return{batch:o=>{let l;e++;try{l=o()}finally{e--,e||u()}return l},batchCalls:o=>(...l)=>{n(()=>{o(...l)})},schedule:n,setNotifyFunction:o=>{s=o},setBatchNotifyFunction:o=>{a=o},setScheduler:o=>{r=o}}}var x=Vl(),Bl=class extends be{#e=!0;#t;#s;constructor(){super(),this.#s=t=>{if(!ae&&window.addEventListener){const e=()=>t(!0),s=()=>t(!1);return window.addEventListener("online",e,!1),window.addEventListener("offline",s,!1),()=>{window.removeEventListener("online",e),window.removeEventListener("offline",s)}}}}onSubscribe(){this.#t||this.setEventListener(this.#s)}onUnsubscribe(){this.hasListeners()||(this.#t?.(),this.#t=void 0)}setEventListener(t){this.#s=t,this.#t?.(),this.#t=t(this.setOnline.bind(this))}setOnline(t){this.#e!==t&&(this.#e=t,this.listeners.forEach(s=>{s(t)}))}isOnline(){return this.#e}},Ce=new Bl;function ql(t){return Math.min(1e3*2**t,3e4)}function Ss(t){return(t??"online")==="online"?Ce.isOnline():!0}var st=class extends Error{constructor(t){super("CancelledError"),this.revert=t?.revert,this.silent=t?.silent}};function vs(t){let e=!1,s=0,a;const r=tt(),n=()=>r.status!=="pending",u=y=>{if(!n()){const f=new st(y);h(f),t.onCancel?.(f)}},o=()=>{e=!0},l=()=>{e=!1},c=()=>yt.isFocused()&&(t.networkMode==="always"||Ce.isOnline())&&t.canRun(),p=()=>Ss(t.networkMode)&&t.canRun(),m=y=>{n()||(a?.(),r.resolve(y))},h=y=>{n()||(a?.(),r.reject(y))},d=()=>new Promise(y=>{a=f=>{(n()||c())&&y(f)},t.onPause?.()}).then(()=>{a=void 0,n()||t.onContinue?.()}),b=()=>{if(n())return;let y;const f=s===0?t.initialPromise:void 0;try{y=f??t.fn()}catch(N){y=Promise.reject(N)}Promise.resolve(y).then(m).catch(N=>{if(n())return;const S=t.retry??(ae?0:3),k=t.retryDelay??ql,T=typeof k=="function"?k(s,N):k,_=S===!0||typeof S=="number"&&s<S||typeof S=="function"&&S(s,N);if(e||!_){h(N);return}s++,t.onFail?.(s,N),xl(T).then(()=>c()?void 0:d()).then(()=>{e?h(N):b()})})};return{promise:r,status:()=>r.status,cancel:u,continue:()=>(a?.(),r),cancelRetry:o,continueRetry:l,canStart:p,start:()=>(p()?b():d().then(b),r)}}var Os=class{#e;destroy(){this.clearGcTimeout()}scheduleGc(){this.clearGcTimeout(),Ye(this.gcTime)&&(this.#e=ee.setTimeout(()=>{this.optionalRemove()},this.gcTime))}updateGcTime(t){this.gcTime=Math.max(this.gcTime||0,t??(ae?1/0:300*1e3))}clearGcTimeout(){this.#e&&(ee.clearTimeout(this.#e),this.#e=void 0)}},Ul=class extends Os{#e;#t;#s;#r;#a;#o;#i;constructor(t){super(),this.#i=!1,this.#o=t.defaultOptions,this.setOptions(t.options),this.observers=[],this.#r=t.client,this.#s=this.#r.getQueryCache(),this.queryKey=t.queryKey,this.queryHash=t.queryHash,this.#e=Vt(this.options),this.state=t.state??this.#e,this.scheduleGc()}get meta(){return this.options.meta}get promise(){return this.#a?.promise}setOptions(t){if(this.options={...this.#o,...t},this.updateGcTime(this.options.gcTime),this.state&&this.state.data===void 0){const e=Vt(this.options);e.data!==void 0&&(this.setData(e.data,{updatedAt:e.dataUpdatedAt,manual:!0}),this.#e=e)}}optionalRemove(){!this.observers.length&&this.state.fetchStatus==="idle"&&this.#s.remove(this)}setData(t,e){const s=et(this.state.data,t,this.options);return this.#n({data:s,type:"success",dataUpdatedAt:e?.updatedAt,manual:e?.manual}),s}setState(t,e){this.#n({type:"setState",state:t,setStateOptions:e})}cancel(t){const e=this.#a?.promise;return this.#a?.cancel(t),e?e.then(R).catch(R):Promise.resolve()}destroy(){super.destroy(),this.cancel({silent:!0})}reset(){this.destroy(),this.setState(this.#e)}isActive(){return this.observers.some(t=>B(t.options.enabled,this)!==!1)}isDisabled(){return this.getObserversCount()>0?!this.isActive():this.options.queryFn===ft||this.state.dataUpdateCount+this.state.errorUpdateCount===0}isStatic(){return this.getObserversCount()>0?this.observers.some(t=>J(t.options.staleTime,this)==="static"):!1}isStale(){return this.getObserversCount()>0?this.observers.some(t=>t.getCurrentResult().isStale):this.state.data===void 0||this.state.isInvalidated}isStaleByTime(t=0){return this.state.data===void 0?!0:t==="static"?!1:this.state.isInvalidated?!0:!Ns(this.state.dataUpdatedAt,t)}onFocus(){this.observers.find(e=>e.shouldFetchOnWindowFocus())?.refetch({cancelRefetch:!1}),this.#a?.continue()}onOnline(){this.observers.find(e=>e.shouldFetchOnReconnect())?.refetch({cancelRefetch:!1}),this.#a?.continue()}addObserver(t){this.observers.includes(t)||(this.observers.push(t),this.clearGcTimeout(),this.#s.notify({type:"observerAdded",query:this,observer:t}))}removeObserver(t){this.observers.includes(t)&&(this.observers=this.observers.filter(e=>e!==t),this.observers.length||(this.#a&&(this.#i?this.#a.cancel({revert:!0}):this.#a.cancelRetry()),this.scheduleGc()),this.#s.notify({type:"observerRemoved",query:this,observer:t}))}getObserversCount(){return this.observers.length}invalidate(){this.state.isInvalidated||this.#n({type:"invalidate"})}async fetch(t,e){if(this.state.fetchStatus!=="idle"&&this.#a?.status()!=="rejected"){if(this.state.data!==void 0&&e?.cancelRefetch)this.cancel({silent:!0});else if(this.#a)return this.#a.continueRetry(),this.#a.promise}if(t&&this.setOptions(t),!this.options.queryFn){const o=this.observers.find(l=>l.options.queryFn);o&&this.setOptions(o.options)}const s=new AbortController,a=o=>{Object.defineProperty(o,"signal",{enumerable:!0,get:()=>(this.#i=!0,s.signal)})},r=()=>{const o=Ts(this.options,e),c=(()=>{const p={client:this.#r,queryKey:this.queryKey,meta:this.meta};return a(p),p})();return this.#i=!1,this.options.persister?this.options.persister(o,c,this):o(c)},u=(()=>{const o={fetchOptions:e,options:this.options,queryKey:this.queryKey,client:this.#r,state:this.state,fetchFn:r};return a(o),o})();this.options.behavior?.onFetch(u,this),this.#t=this.state,(this.state.fetchStatus==="idle"||this.state.fetchMeta!==u.fetchOptions?.meta)&&this.#n({type:"fetch",meta:u.fetchOptions?.meta}),this.#a=vs({initialPromise:e?.initialPromise,fn:u.fetchFn,onCancel:o=>{o instanceof st&&o.revert&&this.setState({...this.#t,fetchStatus:"idle"}),s.abort()},onFail:(o,l)=>{this.#n({type:"failed",failureCount:o,error:l})},onPause:()=>{this.#n({type:"pause"})},onContinue:()=>{this.#n({type:"continue"})},retry:u.options.retry,retryDelay:u.options.retryDelay,networkMode:u.options.networkMode,canRun:()=>!0});try{const o=await this.#a.start();if(o===void 0)throw new Error(`${this.queryHash} data is undefined`);return this.setData(o),this.#s.config.onSuccess?.(o,this),this.#s.config.onSettled?.(o,this.state.error,this),o}catch(o){if(o instanceof st){if(o.silent)return this.#a.promise;if(o.revert){if(this.state.data===void 0)throw o;return this.state.data}}throw this.#n({type:"error",error:o}),this.#s.config.onError?.(o,this),this.#s.config.onSettled?.(this.state.data,o,this),o}finally{this.scheduleGc()}}#n(t){const e=s=>{switch(t.type){case"failed":return{...s,fetchFailureCount:t.failureCount,fetchFailureReason:t.error};case"pause":return{...s,fetchStatus:"paused"};case"continue":return{...s,fetchStatus:"fetching"};case"fetch":return{...s,...ks(s.data,this.options),fetchMeta:t.meta??null};case"success":const a={...s,data:t.data,dataUpdateCount:s.dataUpdateCount+1,dataUpdatedAt:t.dataUpdatedAt??Date.now(),error:null,isInvalidated:!1,status:"success",...!t.manual&&{fetchStatus:"idle",fetchFailureCount:0,fetchFailureReason:null}};return this.#t=t.manual?a:void 0,a;case"error":const r=t.error;return{...s,error:r,errorUpdateCount:s.errorUpdateCount+1,errorUpdatedAt:Date.now(),fetchFailureCount:s.fetchFailureCount+1,fetchFailureReason:r,fetchStatus:"idle",status:"error"};case"invalidate":return{...s,isInvalidated:!0};case"setState":return{...s,...t.state}}};this.state=e(this.state),x.batch(()=>{this.observers.forEach(s=>{s.onQueryUpdate()}),this.#s.notify({query:this,type:"updated",action:t})})}};function ks(t,e){return{fetchFailureCount:0,fetchFailureReason:null,fetchStatus:Ss(e.networkMode)?"fetching":"paused",...t===void 0&&{error:null,status:"pending"}}}function Vt(t){const e=typeof t.initialData=="function"?t.initialData():t.initialData,s=e!==void 0,a=s?typeof t.initialDataUpdatedAt=="function"?t.initialDataUpdatedAt():t.initialDataUpdatedAt:0;return{data:e,dataUpdateCount:0,dataUpdatedAt:s?a??Date.now():0,error:null,errorUpdateCount:0,errorUpdatedAt:0,fetchFailureCount:0,fetchFailureReason:null,fetchMeta:null,isInvalidated:!1,status:s?"success":"pending",fetchStatus:"idle"}}var Hl=class extends be{constructor(t,e){super(),this.options=e,this.#e=t,this.#n=null,this.#i=tt(),this.bindMethods(),this.setOptions(e)}#e;#t=void 0;#s=void 0;#r=void 0;#a;#o;#i;#n;#f;#m;#h;#l;#c;#u;#d=new Set;bindMethods(){this.refetch=this.refetch.bind(this)}onSubscribe(){this.listeners.size===1&&(this.#t.addObserver(this),Bt(this.#t,this.options)?this.#p():this.updateResult(),this.#N())}onUnsubscribe(){this.hasListeners()||this.destroy()}shouldFetchOnReconnect(){return at(this.#t,this.options,this.options.refetchOnReconnect)}shouldFetchOnWindowFocus(){return at(this.#t,this.options,this.options.refetchOnWindowFocus)}destroy(){this.listeners=new Set,this.#w(),this.#T(),this.#t.removeObserver(this)}setOptions(t){const e=this.options,s=this.#t;if(this.options=this.#e.defaultQueryOptions(t),this.options.enabled!==void 0&&typeof this.options.enabled!="boolean"&&typeof this.options.enabled!="function"&&typeof B(this.options.enabled,this.#t)!="boolean")throw new Error("Expected enabled to be a boolean or a callback that returns a boolean");this.#S(),this.#t.setOptions(this.options),e._defaulted&&!Ze(this.options,e)&&this.#e.getQueryCache().notify({type:"observerOptionsUpdated",query:this.#t,observer:this});const a=this.hasListeners();a&&qt(this.#t,s,this.options,e)&&this.#p(),this.updateResult(),a&&(this.#t!==s||B(this.options.enabled,this.#t)!==B(e.enabled,this.#t)||J(this.options.staleTime,this.#t)!==J(e.staleTime,this.#t))&&this.#y();const r=this.#g();a&&(this.#t!==s||B(this.options.enabled,this.#t)!==B(e.enabled,this.#t)||r!==this.#u)&&this.#b(r)}getOptimisticResult(t){const e=this.#e.getQueryCache().build(this.#e,t),s=this.createResult(e,t);return Gl(this,s)&&(this.#r=s,this.#o=this.options,this.#a=this.#t.state),s}getCurrentResult(){return this.#r}trackResult(t,e){return new Proxy(t,{get:(s,a)=>(this.trackProp(a),e?.(a),a==="promise"&&!this.options.experimental_prefetchInRender&&this.#i.status==="pending"&&this.#i.reject(new Error("experimental_prefetchInRender feature flag is not enabled")),Reflect.get(s,a))})}trackProp(t){this.#d.add(t)}getCurrentQuery(){return this.#t}refetch({...t}={}){return this.fetch({...t})}fetchOptimistic(t){const e=this.#e.defaultQueryOptions(t),s=this.#e.getQueryCache().build(this.#e,e);return s.fetch().then(()=>this.createResult(s,e))}fetch(t){return this.#p({...t,cancelRefetch:t.cancelRefetch??!0}).then(()=>(this.updateResult(),this.#r))}#p(t){this.#S();let e=this.#t.fetch(this.options,t);return t?.throwOnError||(e=e.catch(R)),e}#y(){this.#w();const t=J(this.options.staleTime,this.#t);if(ae||this.#r.isStale||!Ye(t))return;const s=Ns(this.#r.dataUpdatedAt,t)+1;this.#l=ee.setTimeout(()=>{this.#r.isStale||this.updateResult()},s)}#g(){return(typeof this.options.refetchInterval=="function"?this.options.refetchInterval(this.#t):this.options.refetchInterval)??!1}#b(t){this.#T(),this.#u=t,!(ae||B(this.options.enabled,this.#t)===!1||!Ye(this.#u)||this.#u===0)&&(this.#c=ee.setInterval(()=>{(this.options.refetchIntervalInBackground||yt.isFocused())&&this.#p()},this.#u))}#N(){this.#y(),this.#b(this.#g())}#w(){this.#l&&(ee.clearTimeout(this.#l),this.#l=void 0)}#T(){this.#c&&(ee.clearInterval(this.#c),this.#c=void 0)}createResult(t,e){const s=this.#t,a=this.options,r=this.#r,n=this.#a,u=this.#o,l=t!==s?t.state:this.#s,{state:c}=t;let p={...c},m=!1,h;if(e._optimisticResults){const I=this.hasListeners(),$=!I&&Bt(t,e),F=I&&qt(t,s,e,a);($||F)&&(p={...p,...ks(c.data,t.options)}),e._optimisticResults==="isRestoring"&&(p.fetchStatus="idle")}let{error:d,errorUpdatedAt:b,status:y}=p;h=p.data;let f=!1;if(e.placeholderData!==void 0&&h===void 0&&y==="pending"){let I;r?.isPlaceholderData&&e.placeholderData===u?.placeholderData?(I=r.data,f=!0):I=typeof e.placeholderData=="function"?e.placeholderData(this.#h?.state.data,this.#h):e.placeholderData,I!==void 0&&(y="success",h=et(r?.data,I,e),m=!0)}if(e.select&&h!==void 0&&!f)if(r&&h===n?.data&&e.select===this.#f)h=this.#m;else try{this.#f=e.select,h=e.select(h),h=et(r?.data,h,e),this.#m=h,this.#n=null}catch(I){this.#n=I}this.#n&&(d=this.#n,h=this.#m,b=Date.now(),y="error");const N=p.fetchStatus==="fetching",S=y==="pending",k=y==="error",T=S&&N,_=h!==void 0,A={status:y,fetchStatus:p.fetchStatus,isPending:S,isSuccess:y==="success",isError:k,isInitialLoading:T,isLoading:T,data:h,dataUpdatedAt:p.dataUpdatedAt,error:d,errorUpdatedAt:b,failureCount:p.fetchFailureCount,failureReason:p.fetchFailureReason,errorUpdateCount:p.errorUpdateCount,isFetched:p.dataUpdateCount>0||p.errorUpdateCount>0,isFetchedAfterMount:p.dataUpdateCount>l.dataUpdateCount||p.errorUpdateCount>l.errorUpdateCount,isFetching:N,isRefetching:N&&!S,isLoadingError:k&&!_,isPaused:p.fetchStatus==="paused",isPlaceholderData:m,isRefetchError:k&&_,isStale:gt(t,e),refetch:this.refetch,promise:this.#i,isEnabled:B(e.enabled,t)!==!1};if(this.options.experimental_prefetchInRender){const I=K=>{A.status==="error"?K.reject(A.error):A.data!==void 0&&K.resolve(A.data)},$=()=>{const K=this.#i=A.promise=tt();I(K)},F=this.#i;switch(F.status){case"pending":t.queryHash===s.queryHash&&I(F);break;case"fulfilled":(A.status==="error"||A.data!==F.value)&&$();break;case"rejected":(A.status!=="error"||A.error!==F.reason)&&$();break}}return A}updateResult(){const t=this.#r,e=this.createResult(this.#t,this.options);if(this.#a=this.#t.state,this.#o=this.options,this.#a.data!==void 0&&(this.#h=this.#t),Ze(e,t))return;this.#r=e;const s=()=>{if(!t)return!0;const{notifyOnChangeProps:a}=this.options,r=typeof a=="function"?a():a;if(r==="all"||!r&&!this.#d.size)return!0;const n=new Set(r??this.#d);return this.options.throwOnError&&n.add("error"),Object.keys(this.#r).some(u=>{const o=u;return this.#r[o]!==t[o]&&n.has(o)})};this.#v({listeners:s()})}#S(){const t=this.#e.getQueryCache().build(this.#e,this.options);if(t===this.#t)return;const e=this.#t;this.#t=t,this.#s=t.state,this.hasListeners()&&(e?.removeObserver(this),t.addObserver(this))}onQueryUpdate(){this.updateResult(),this.hasListeners()&&this.#N()}#v(t){x.batch(()=>{t.listeners&&this.listeners.forEach(e=>{e(this.#r)}),this.#e.getQueryCache().notify({query:this.#t,type:"observerResultsUpdated"})})}};function Wl(t,e){return B(e.enabled,t)!==!1&&t.state.data===void 0&&!(t.state.status==="error"&&e.retryOnMount===!1)}function Bt(t,e){return Wl(t,e)||t.state.data!==void 0&&at(t,e,e.refetchOnMount)}function at(t,e,s){if(B(e.enabled,t)!==!1&&J(e.staleTime,t)!=="static"){const a=typeof s=="function"?s(t):s;return a==="always"||a!==!1&&gt(t,e)}return!1}function qt(t,e,s,a){return(t!==e||B(a.enabled,t)===!1)&&(!s.suspense||t.state.status!=="error")&&gt(t,s)}function gt(t,e){return B(e.enabled,t)!==!1&&t.isStaleByTime(J(e.staleTime,t))}function Gl(t,e){return!Ze(t.getCurrentResult(),e)}function Ut(t){return{onFetch:(e,s)=>{const a=e.options,r=e.fetchOptions?.meta?.fetchMore?.direction,n=e.state.data?.pages||[],u=e.state.data?.pageParams||[];let o={pages:[],pageParams:[]},l=0;const c=async()=>{let p=!1;const m=b=>{Object.defineProperty(b,"signal",{enumerable:!0,get:()=>(e.signal.aborted?p=!0:e.signal.addEventListener("abort",()=>{p=!0}),e.signal)})},h=Ts(e.options,e.fetchOptions),d=async(b,y,f)=>{if(p)return Promise.reject();if(y==null&&b.pages.length)return Promise.resolve(b);const S=(()=>{const j={client:e.client,queryKey:e.queryKey,pageParam:y,direction:f?"backward":"forward",meta:e.options.meta};return m(j),j})(),k=await h(S),{maxPages:T}=e.options,_=f?Fl:$l;return{pages:_(b.pages,k,T),pageParams:_(b.pageParams,y,T)}};if(r&&n.length){const b=r==="backward",y=b?Ql:Ht,f={pages:n,pageParams:u},N=y(a,f);o=await d(f,N,b)}else{const b=t??n.length;do{const y=l===0?u[0]??a.initialPageParam:Ht(a,o);if(l>0&&y==null)break;o=await d(o,y),l++}while(l<b)}return o};e.options.persister?e.fetchFn=()=>e.options.persister?.(c,{client:e.client,queryKey:e.queryKey,meta:e.options.meta,signal:e.signal},s):e.fetchFn=c}}}function Ht(t,{pages:e,pageParams:s}){const a=e.length-1;return e.length>0?t.getNextPageParam(e[a],e,s[a],s):void 0}function Ql(t,{pages:e,pageParams:s}){return e.length>0?t.getPreviousPageParam?.(e[0],e,s[0],s):void 0}var Kl=class extends Os{#e;#t;#s;#r;constructor(t){super(),this.#e=t.client,this.mutationId=t.mutationId,this.#s=t.mutationCache,this.#t=[],this.state=t.state||Jl(),this.setOptions(t.options),this.scheduleGc()}setOptions(t){this.options=t,this.updateGcTime(this.options.gcTime)}get meta(){return this.options.meta}addObserver(t){this.#t.includes(t)||(this.#t.push(t),this.clearGcTimeout(),this.#s.notify({type:"observerAdded",mutation:this,observer:t}))}removeObserver(t){this.#t=this.#t.filter(e=>e!==t),this.scheduleGc(),this.#s.notify({type:"observerRemoved",mutation:this,observer:t})}optionalRemove(){this.#t.length||(this.state.status==="pending"?this.scheduleGc():this.#s.remove(this))}continue(){return this.#r?.continue()??this.execute(this.state.variables)}async execute(t){const e=()=>{this.#a({type:"continue"})},s={client:this.#e,meta:this.options.meta,mutationKey:this.options.mutationKey};this.#r=vs({fn:()=>this.options.mutationFn?this.options.mutationFn(t,s):Promise.reject(new Error("No mutationFn found")),onFail:(n,u)=>{this.#a({type:"failed",failureCount:n,error:u})},onPause:()=>{this.#a({type:"pause"})},onContinue:e,retry:this.options.retry??0,retryDelay:this.options.retryDelay,networkMode:this.options.networkMode,canRun:()=>this.#s.canRun(this)});const a=this.state.status==="pending",r=!this.#r.canStart();try{if(a)e();else{this.#a({type:"pending",variables:t,isPaused:r}),await this.#s.config.onMutate?.(t,this,s);const u=await this.options.onMutate?.(t,s);u!==this.state.context&&this.#a({type:"pending",context:u,variables:t,isPaused:r})}const n=await this.#r.start();return await this.#s.config.onSuccess?.(n,t,this.state.context,this,s),await this.options.onSuccess?.(n,t,this.state.context,s),await this.#s.config.onSettled?.(n,null,this.state.variables,this.state.context,this,s),await this.options.onSettled?.(n,null,t,this.state.context,s),this.#a({type:"success",data:n}),n}catch(n){try{throw await this.#s.config.onError?.(n,t,this.state.context,this,s),await this.options.onError?.(n,t,this.state.context,s),await this.#s.config.onSettled?.(void 0,n,this.state.variables,this.state.context,this,s),await this.options.onSettled?.(void 0,n,t,this.state.context,s),n}finally{this.#a({type:"error",error:n})}}finally{this.#s.runNext(this)}}#a(t){const e=s=>{switch(t.type){case"failed":return{...s,failureCount:t.failureCount,failureReason:t.error};case"pause":return{...s,isPaused:!0};case"continue":return{...s,isPaused:!1};case"pending":return{...s,context:t.context,data:void 0,failureCount:0,failureReason:null,error:null,isPaused:t.isPaused,status:"pending",variables:t.variables,submittedAt:Date.now()};case"success":return{...s,data:t.data,failureCount:0,failureReason:null,error:null,status:"success",isPaused:!1};case"error":return{...s,data:void 0,error:t.error,failureCount:s.failureCount+1,failureReason:t.error,isPaused:!1,status:"error"}}};this.state=e(this.state),x.batch(()=>{this.#t.forEach(s=>{s.onMutationUpdate(t)}),this.#s.notify({mutation:this,type:"updated",action:t})})}};function Jl(){return{context:void 0,data:void 0,error:null,failureCount:0,failureReason:null,isPaused:!1,status:"idle",variables:void 0,submittedAt:0}}var Xl=class extends be{constructor(t={}){super(),this.config=t,this.#e=new Set,this.#t=new Map,this.#s=0}#e;#t;#s;build(t,e,s){const a=new Kl({client:t,mutationCache:this,mutationId:++this.#s,options:t.defaultMutationOptions(e),state:s});return this.add(a),a}add(t){this.#e.add(t);const e=Te(t);if(typeof e=="string"){const s=this.#t.get(e);s?s.push(t):this.#t.set(e,[t])}this.notify({type:"added",mutation:t})}remove(t){if(this.#e.delete(t)){const e=Te(t);if(typeof e=="string"){const s=this.#t.get(e);if(s)if(s.length>1){const a=s.indexOf(t);a!==-1&&s.splice(a,1)}else s[0]===t&&this.#t.delete(e)}}this.notify({type:"removed",mutation:t})}canRun(t){const e=Te(t);if(typeof e=="string"){const a=this.#t.get(e)?.find(r=>r.state.status==="pending");return!a||a===t}else return!0}runNext(t){const e=Te(t);return typeof e=="string"?this.#t.get(e)?.find(a=>a!==t&&a.state.isPaused)?.continue()??Promise.resolve():Promise.resolve()}clear(){x.batch(()=>{this.#e.forEach(t=>{this.notify({type:"removed",mutation:t})}),this.#e.clear(),this.#t.clear()})}getAll(){return Array.from(this.#e)}find(t){const e={exact:!0,...t};return this.getAll().find(s=>zt(e,s))}findAll(t={}){return this.getAll().filter(e=>zt(t,e))}notify(t){x.batch(()=>{this.listeners.forEach(e=>{e(t)})})}resumePausedMutations(){const t=this.getAll().filter(e=>e.state.isPaused);return x.batch(()=>Promise.all(t.map(e=>e.continue().catch(R))))}};function Te(t){return t.options.scope?.id}var Yl=class extends be{constructor(t={}){super(),this.config=t,this.#e=new Map}#e;build(t,e,s){const a=e.queryKey,r=e.queryHash??dt(a,e);let n=this.get(r);return n||(n=new Ul({client:t,queryKey:a,queryHash:r,options:t.defaultQueryOptions(e),state:s,defaultOptions:t.getQueryDefaults(a)}),this.add(n)),n}add(t){this.#e.has(t.queryHash)||(this.#e.set(t.queryHash,t),this.notify({type:"added",query:t}))}remove(t){const e=this.#e.get(t.queryHash);e&&(t.destroy(),e===t&&this.#e.delete(t.queryHash),this.notify({type:"removed",query:t}))}clear(){x.batch(()=>{this.getAll().forEach(t=>{this.remove(t)})})}get(t){return this.#e.get(t)}getAll(){return[...this.#e.values()]}find(t){const e={exact:!0,...t};return this.getAll().find(s=>Ft(e,s))}findAll(t={}){const e=this.getAll();return Object.keys(t).length>0?e.filter(s=>Ft(t,s)):e}notify(t){x.batch(()=>{this.listeners.forEach(e=>{e(t)})})}onFocus(){x.batch(()=>{this.getAll().forEach(t=>{t.onFocus()})})}onOnline(){x.batch(()=>{this.getAll().forEach(t=>{t.onOnline()})})}},Zl=class{#e;#t;#s;#r;#a;#o;#i;#n;constructor(t={}){this.#e=t.queryCache||new Yl,this.#t=t.mutationCache||new Xl,this.#s=t.defaultOptions||{},this.#r=new Map,this.#a=new Map,this.#o=0}mount(){this.#o++,this.#o===1&&(this.#i=yt.subscribe(async t=>{t&&(await this.resumePausedMutations(),this.#e.onFocus())}),this.#n=Ce.subscribe(async t=>{t&&(await this.resumePausedMutations(),this.#e.onOnline())}))}unmount(){this.#o--,this.#o===0&&(this.#i?.(),this.#i=void 0,this.#n?.(),this.#n=void 0)}isFetching(t){return this.#e.findAll({...t,fetchStatus:"fetching"}).length}isMutating(t){return this.#t.findAll({...t,status:"pending"}).length}getQueryData(t){const e=this.defaultQueryOptions({queryKey:t});return this.#e.get(e.queryHash)?.state.data}ensureQueryData(t){const e=this.defaultQueryOptions(t),s=this.#e.build(this,e),a=s.state.data;return a===void 0?this.fetchQuery(t):(t.revalidateIfStale&&s.isStaleByTime(J(e.staleTime,s))&&this.prefetchQuery(e),Promise.resolve(a))}getQueriesData(t){return this.#e.findAll(t).map(({queryKey:e,state:s})=>{const a=s.data;return[e,a]})}setQueryData(t,e,s){const a=this.defaultQueryOptions({queryKey:t}),n=this.#e.get(a.queryHash)?.state.data,u=Pl(e,n);if(u!==void 0)return this.#e.build(this,a).setData(u,{...s,manual:!0})}setQueriesData(t,e,s){return x.batch(()=>this.#e.findAll(t).map(({queryKey:a})=>[a,this.setQueryData(a,e,s)]))}getQueryState(t){const e=this.defaultQueryOptions({queryKey:t});return this.#e.get(e.queryHash)?.state}removeQueries(t){const e=this.#e;x.batch(()=>{e.findAll(t).forEach(s=>{e.remove(s)})})}resetQueries(t,e){const s=this.#e;return x.batch(()=>(s.findAll(t).forEach(a=>{a.reset()}),this.refetchQueries({type:"active",...t},e)))}cancelQueries(t,e={}){const s={revert:!0,...e},a=x.batch(()=>this.#e.findAll(t).map(r=>r.cancel(s)));return Promise.all(a).then(R).catch(R)}invalidateQueries(t,e={}){return x.batch(()=>(this.#e.findAll(t).forEach(s=>{s.invalidate()}),t?.refetchType==="none"?Promise.resolve():this.refetchQueries({...t,type:t?.refetchType??t?.type??"active"},e)))}refetchQueries(t,e={}){const s={...e,cancelRefetch:e.cancelRefetch??!0},a=x.batch(()=>this.#e.findAll(t).filter(r=>!r.isDisabled()&&!r.isStatic()).map(r=>{let n=r.fetch(void 0,s);return s.throwOnError||(n=n.catch(R)),r.state.fetchStatus==="paused"?Promise.resolve():n}));return Promise.all(a).then(R)}fetchQuery(t){const e=this.defaultQueryOptions(t);e.retry===void 0&&(e.retry=!1);const s=this.#e.build(this,e);return s.isStaleByTime(J(e.staleTime,s))?s.fetch(e):Promise.resolve(s.state.data)}prefetchQuery(t){return this.fetchQuery(t).then(R).catch(R)}fetchInfiniteQuery(t){return t.behavior=Ut(t.pages),this.fetchQuery(t)}prefetchInfiniteQuery(t){return this.fetchInfiniteQuery(t).then(R).catch(R)}ensureInfiniteQueryData(t){return t.behavior=Ut(t.pages),this.ensureQueryData(t)}resumePausedMutations(){return Ce.isOnline()?this.#t.resumePausedMutations():Promise.resolve()}getQueryCache(){return this.#e}getMutationCache(){return this.#t}getDefaultOptions(){return this.#s}setDefaultOptions(t){this.#s=t}setQueryDefaults(t,e){this.#r.set(he(t),{queryKey:t,defaultOptions:e})}getQueryDefaults(t){const e=[...this.#r.values()],s={};return e.forEach(a=>{de(t,a.queryKey)&&Object.assign(s,a.defaultOptions)}),s}setMutationDefaults(t,e){this.#a.set(he(t),{mutationKey:t,defaultOptions:e})}getMutationDefaults(t){const e=[...this.#a.values()],s={};return e.forEach(a=>{de(t,a.mutationKey)&&Object.assign(s,a.defaultOptions)}),s}defaultQueryOptions(t){if(t._defaulted)return t;const e={...this.#s.queries,...this.getQueryDefaults(t.queryKey),...t,_defaulted:!0};return e.queryHash||(e.queryHash=dt(e.queryKey,e)),e.refetchOnReconnect===void 0&&(e.refetchOnReconnect=e.networkMode!=="always"),e.throwOnError===void 0&&(e.throwOnError=!!e.suspense),!e.networkMode&&e.persister&&(e.networkMode="offlineFirst"),e.queryFn===ft&&(e.enabled=!1),e}defaultMutationOptions(t){return t?._defaulted?t:{...this.#s.mutations,...t?.mutationKey&&this.getMutationDefaults(t.mutationKey),...t,_defaulted:!0}}clear(){this.#e.clear(),this.#t.clear()}},_s=q.createContext(void 0),Ml=t=>{const e=q.useContext(_s);if(!e)throw new Error("No QueryClient set, use QueryClientProvider to set one");return e},ec=({client:t,children:e})=>(q.useEffect(()=>(t.mount(),()=>{t.unmount()}),[t]),g.jsx(_s.Provider,{value:t,children:e})),Es=q.createContext(!1),tc=()=>q.useContext(Es);Es.Provider;function sc(){let t=!1;return{clearReset:()=>{t=!1},reset:()=>{t=!0},isReset:()=>t}}var ac=q.createContext(sc()),rc=()=>q.useContext(ac),nc=(t,e)=>{(t.suspense||t.throwOnError||t.experimental_prefetchInRender)&&(e.isReset()||(t.retryOnMount=!1))},ic=t=>{q.useEffect(()=>{t.clearReset()},[t])},oc=({result:t,errorResetBoundary:e,throwOnError:s,query:a,suspense:r})=>t.isError&&!e.isReset()&&!t.isFetching&&a&&(r&&t.data===void 0||zl(s,[t.error,a])),uc=t=>{if(t.suspense){const s=r=>r==="static"?r:Math.max(r??1e3,1e3),a=t.staleTime;t.staleTime=typeof a=="function"?(...r)=>s(a(...r)):s(a),typeof t.gcTime=="number"&&(t.gcTime=Math.max(t.gcTime,1e3))}},lc=(t,e)=>t.isLoading&&t.isFetching&&!e,cc=(t,e)=>t?.suspense&&e.isPending,Wt=(t,e,s)=>e.fetchOptimistic(t).catch(()=>{s.clearReset()});function pc(t,e,s){const a=tc(),r=rc(),n=Ml(),u=n.defaultQueryOptions(t);n.getDefaultOptions().queries?._experimental_beforeQuery?.(u),u._optimisticResults=a?"isRestoring":"optimistic",uc(u),nc(u,r),ic(r);const o=!n.getQueryCache().get(u.queryHash),[l]=q.useState(()=>new e(n,u)),c=l.getOptimisticResult(u),p=!a&&t.subscribed!==!1;if(q.useSyncExternalStore(q.useCallback(m=>{const h=p?l.subscribe(x.batchCalls(m)):R;return l.updateResult(),h},[l,p]),()=>l.getCurrentResult(),()=>l.getCurrentResult()),q.useEffect(()=>{l.setOptions(u)},[u,l]),cc(u,c))throw Wt(u,l,r);if(oc({result:c,errorResetBoundary:r,throwOnError:u.throwOnError,query:n.getQueryCache().get(u.queryHash),suspense:u.suspense}))throw c.error;return n.getDefaultOptions().queries?._experimental_afterQuery?.(u,c),u.experimental_prefetchInRender&&!ae&&lc(c,a)&&(o?Wt(u,l,r):n.getQueryCache().get(u.queryHash)?.promise)?.catch(R).finally(()=>{l.updateResult()}),u.notifyOnChangeProps?c:l.trackResult(c)}function mc(t,e){return pc(t,Hl)}const hc=new Zl;function dc({children:t}){return g.jsx(ec,{client:hc,children:t})}const fc="https://www.kaggle.com/models/google/inception-v3/TfJs/classification/2",Gt="indexeddb://inception-v3",Se=299,yc=100,As=2,gc=["background","tench","goldfish","great white shark","tiger shark","hammerhead","electric ray","stingray","cock","hen","ostrich","brambling","goldfinch","house finch","junco","indigo bunting","robin","bulbul","jay","magpie","chickadee","water ouzel","kite","bald eagle","vulture","great grey owl","European fire salamander","common newt","eft","spotted salamander","axolotl","bullfrog","tree frog","tailed frog","loggerhead","leatherback turtle","mud turtle","terrapin","box turtle","banded gecko","common iguana","American chameleon","whiptail","agama","frilled lizard","alligator lizard","Gila monster","green lizard","African chameleon","Komodo dragon","African crocodile","American alligator","triceratops","thunder snake","ringneck snake","hognose snake","green snake","king snake","garter snake","water snake","vine snake","night snake","boa constrictor","rock python","Indian cobra","green mamba","sea snake","horned viper","diamondback","sidewinder","trilobite","harvestman","scorpion","black and gold garden spider","barn spider","garden spider","black widow","tarantula","wolf spider","tick","centipede","black grouse","ptarmigan","ruffed grouse","prairie chicken","peacock","quail","partridge","African grey","macaw","sulphur-crested cockatoo","lorikeet","coucal","bee eater","hornbill","hummingbird","jacamar","toucan","drake","red-breasted merganser","goose","black swan","tusker","echidna","platypus","wallaby","koala","wombat","jellyfish","sea anemone","brain coral","flatworm","nematode","conch","snail","slug","sea slug","chiton","chambered nautilus","Dungeness crab","rock crab","fiddler crab","king crab","American lobster","spiny lobster","crayfish","hermit crab","isopod","white stork","black stork","spoonbill","flamingo","little blue heron","American egret","bittern","crane","limpkin","European gallinule","American coot","bustard","ruddy turnstone","red-backed sandpiper","redshank","dowitcher","oystercatcher","pelican","king penguin","albatross","grey whale","killer whale","dugong","sea lion","Chihuahua","Japanese spaniel","Maltese dog","Pekinese","Shih-Tzu","Blenheim spaniel","papillon","toy terrier","Rhodesian ridgeback","Afghan hound","basset","beagle","bloodhound","bluetick","black-and-tan coonhound","Walker hound","English foxhound","redbone","borzoi","Irish wolfhound","Italian greyhound","whippet","Ibizan hound","Norwegian elkhound","otterhound","Saluki","Scottish deerhound","Weimaraner","Staffordshire bullterrier","American Staffordshire terrier","Bedlington terrier","Border terrier","Kerry blue terrier","Irish terrier","Norfolk terrier","Norwich terrier","Yorkshire terrier","wire-haired fox terrier","Lakeland terrier","Sealyham terrier","Airedale","cairn","Australian terrier","Dandie Dinmont","Boston bull","miniature schnauzer","giant schnauzer","standard schnauzer","Scotch terrier","Tibetan terrier","silky terrier","soft-coated wheaten terrier","West Highland white terrier","Lhasa","flat-coated retriever","curly-coated retriever","golden retriever","Labrador retriever","Chesapeake Bay retriever","German short-haired pointer","vizsla","English setter","Irish setter","Gordon setter","Brittany spaniel","clumber","English springer","Welsh springer spaniel","cocker spaniel","Sussex spaniel","Irish water spaniel","kuvasz","schipperke","groenendael","malinois","briard","kelpie","komondor","Old English sheepdog","Shetland sheepdog","collie","Border collie","Bouvier des Flandres","Rottweiler","German shepherd","Doberman","miniature pinscher","Greater Swiss Mountain dog","Bernese mountain dog","Appenzeller","EntleBucher","boxer","bull mastiff","Tibetan mastiff","French bulldog","Great Dane","Saint Bernard","Eskimo dog","malamute","Siberian husky","dalmatian","affenpinscher","basenji","pug","Leonberg","Newfoundland","Great Pyrenees","Samoyed","Pomeranian","chow","keeshond","Brabancon griffon","Pembroke","Cardigan","toy poodle","miniature poodle","standard poodle","Mexican hairless","timber wolf","white wolf","red wolf","coyote","dingo","dhole","African hunting dog","hyena","red fox","kit fox","Arctic fox","grey fox","tabby","tiger cat","Persian cat","Siamese cat","Egyptian cat","cougar","lynx","leopard","snow leopard","jaguar","lion","tiger","cheetah","brown bear","American black bear","ice bear","sloth bear","mongoose","meerkat","tiger beetle","ladybug","ground beetle","long-horned beetle","leaf beetle","dung beetle","rhinoceros beetle","weevil","fly","bee","ant","grasshopper","cricket","walking stick","cockroach","mantis","cicada","leafhopper","lacewing","dragonfly","damselfly","admiral","ringlet","monarch","cabbage butterfly","sulphur butterfly","lycaenid","starfish","sea urchin","sea cucumber","wood rabbit","hare","Angora","hamster","porcupine","fox squirrel","marmot","beaver","guinea pig","sorrel","zebra","hog","wild boar","warthog","hippopotamus","ox","water buffalo","bison","ram","bighorn","ibex","hartebeest","impala","gazelle","Arabian camel","llama","weasel","mink","polecat","black-footed ferret","otter","skunk","badger","armadillo","three-toed sloth","orangutan","gorilla","chimpanzee","gibbon","siamang","guenon","patas","baboon","macaque","langur","colobus","proboscis monkey","marmoset","capuchin","howler monkey","titi","spider monkey","squirrel monkey","Madagascar cat","indri","Indian elephant","African elephant","lesser panda","giant panda","barracouta","eel","coho","rock beauty","anemone fish","sturgeon","gar","lionfish","puffer","abacus","abaya","academic gown","accordion","acoustic guitar","aircraft carrier","airliner","airship","altar","ambulance","amphibian","analog clock","apiary","apron","ashcan","assault rifle","backpack","bakery","balance beam","balloon","ballpoint","Band Aid","banjo","bannister","barbell","barber chair","barbershop","barn","barometer","barrel","barrow","baseball","basketball","bassinet","bassoon","bathing cap","bath towel","bathtub","beach wagon","beacon","beaker","bearskin","beer bottle","beer glass","bell cote","bib","bicycle-built-for-two","bikini","binder","binoculars","birdhouse","boathouse","bobsled","bolo tie","bonnet","bookcase","bookshop","bottlecap","bow","bow tie","brass","brassiere","breakwater","breastplate","broom","bucket","buckle","bulletproof vest","bullet train","butcher shop","cab","caldron","candle","cannon","canoe","can opener","cardigan","car mirror","carousel","carpenter's kit","carton","car wheel","cash machine","cassette","cassette player","castle","catamaran","CD player","cello","cellular telephone","chain","chainlink fence","chain mail","chain saw","chest","chiffonier","chime","china cabinet","Christmas stocking","church","cinema","cleaver","cliff dwelling","cloak","clog","cocktail shaker","coffee mug","coffeepot","coil","combination lock","computer keyboard","confectionery","container ship","convertible","corkscrew","cornet","cowboy boot","cowboy hat","cradle","crane","crash helmet","crate","crib","Crock Pot","croquet ball","crutch","cuirass","dam","desk","desktop computer","dial telephone","diaper","digital clock","digital watch","dining table","dishrag","dishwasher","disk brake","dock","dogsled","dome","doormat","drilling platform","drum","drumstick","dumbbell","Dutch oven","electric fan","electric guitar","electric locomotive","entertainment center","envelope","espresso maker","face powder","feather boa","file","fireboat","fire engine","fire screen","flagpole","flute","folding chair","football helmet","forklift","fountain","fountain pen","four-poster","freight car","French horn","frying pan","fur coat","garbage truck","gasmask","gas pump","goblet","go-kart","golf ball","golfcart","gondola","gong","gown","grand piano","greenhouse","grille","grocery store","guillotine","hair slide","hair spray","half track","hammer","hamper","hand blower","hand-held computer","handkerchief","hard disc","harmonica","harp","harvester","hatchet","holster","home theater","honeycomb","hook","hoopskirt","horizontal bar","horse cart","hourglass","iPod","iron","jack-o'-lantern","jean","jeep","jersey","jigsaw puzzle","jinrikisha","joystick","kimono","knee pad","knot","lab coat","ladle","lampshade","laptop","lawn mower","lens cap","letter opener","library","lifeboat","lighter","limousine","liner","lipstick","Loafer","lotion","loudspeaker","loupe","lumbermill","magnetic compass","mailbag","mailbox","maillot","maillot","manhole cover","maraca","marimba","mask","matchstick","maypole","maze","measuring cup","medicine chest","megalith","microphone","microwave","military uniform","milk can","minibus","miniskirt","minivan","missile","mitten","mixing bowl","mobile home","Model T","modem","monastery","monitor","moped","mortar","mortarboard","mosque","mosquito net","motor scooter","mountain bike","mountain tent","mouse","mousetrap","moving van","muzzle","nail","neck brace","necklace","nipple","notebook","obelisk","oboe","ocarina","odometer","oil filter","organ","oscilloscope","overskirt","oxcart","oxygen mask","packet","paddle","paddlewheel","padlock","paintbrush","pajama","palace","panpipe","paper towel","parachute","parallel bars","park bench","parking meter","passenger car","patio","pay-phone","pedestal","pencil box","pencil sharpener","perfume","Petri dish","photocopier","pick","pickelhaube","picket fence","pickup","pier","piggy bank","pill bottle","pillow","ping-pong ball","pinwheel","pirate","pitcher","plane","planetarium","plastic bag","plate rack","plow","plunger","Polaroid camera","pole","police van","poncho","pool table","pop bottle","pot","potter's wheel","power drill","prayer rug","printer","prison","projectile","projector","puck","punching bag","purse","quill","quilt","racer","racket","radiator","radio","radio telescope","rain barrel","recreational vehicle","reel","reflex camera","refrigerator","remote control","restaurant","revolver","rifle","rocking chair","rotisserie","rubber eraser","rugby ball","rule","running shoe","safe","safety pin","saltshaker","sandal","sarong","sax","scabbard","scale","school bus","schooner","scoreboard","screen","screw","screwdriver","seat belt","sewing machine","shield","shoe shop","shoji","shopping basket","shopping cart","shovel","shower cap","shower curtain","ski","ski mask","sleeping bag","slide rule","sliding door","slot","snorkel","snowmobile","snowplow","soap dispenser","soccer ball","sock","solar dish","sombrero","soup bowl","space bar","space heater","space shuttle","spatula","speedboat","spider web","spindle","sports car","spotlight","stage","steam locomotive","steel arch bridge","steel drum","stethoscope","stole","stone wall","stopwatch","stove","strainer","streetcar","stretcher","studio couch","stupa","submarine","suit","sundial","sunglass","sunglasses","sunscreen","suspension bridge","swab","sweatshirt","swimming trunks","swing","switch","syringe","table lamp","tank","tape player","teapot","teddy","television","tennis ball","thatch","theater curtain","thimble","thresher","throne","tile roof","toaster","tobacco shop","toilet seat","torch","totem pole","tow truck","toyshop","tractor","trailer truck","tray","trench coat","tricycle","trimaran","tripod","triumphal arch","trolleybus","trombone","tub","turnstile","typewriter keyboard","umbrella","unicycle","upright","vacuum","vase","vault","velvet","vending machine","vestment","viaduct","violin","volleyball","waffle iron","wall clock","wallet","wardrobe","warplane","washbasin","washer","water bottle","water jug","water tower","whiskey jug","whistle","wig","window screen","window shade","Windsor tie","wine bottle","wing","wok","wooden spoon","wool","worm fence","wreck","yawl","yurt","web site","comic book","crossword puzzle","street sign","traffic light","book jacket","menu","plate","guacamole","consomme","hot pot","trifle","ice cream","ice lolly","French loaf","bagel","pretzel","cheeseburger","hotdog","mashed potato","head cabbage","broccoli","cauliflower","zucchini","spaghetti squash","acorn squash","butternut squash","cucumber","artichoke","bell pepper","cardoon","mushroom","Granny Smith","strawberry","orange","lemon","fig","pineapple","banana","jackfruit","custard apple","pomegranate","hay","carbonara","chocolate sauce","dough","meat loaf","pizza","potpie","burrito","red wine","espresso","cup","eggnog","alp","bubble","cliff","coral reef","geyser","lakeside","promontory","sandbar","seashore","valley","volcano","ballplayer","groom","scuba diver","rapeseed","daisy","yellow lady's slipper","corn","acorn","hip","buckeye","coral fungus","agaric","gyromitra","stinkhorn","earthstar","hen-of-the-woods","bolete","ear","toilet tissue"];function bc({img:t,model:e,onSuccess:s}){return U(()=>{const a=mi(t),r=os.resizeBilinear(a,[Se,Se],!0).div(255).reshape([1,Se,Se,3]),n=e.predict(r);if(Array.isArray(n)||!(n instanceof pe))throw new Error("Something went wrong. Unexpected result type");const{indices:u,values:o}=us(n,yc),l=o.asType("int32").dataSync(),c=Array.from(u.dataSync());s(c.reduce((p,m,h)=>l[h]<=As?p:[...p,{label:gc[m]??"Unknown",confidence:l[h]}],[]))})}function Nc({className:t,data:e,...s}){return e.length?g.jsx("div",{className:_e("overflow-x-auto rounded-box border border-base-content/5 bg-base-100 p-1 max-h-60",t),children:g.jsxs("table",{className:"table w-full table-pin-rows",...s,children:[g.jsx("thead",{children:g.jsxs("tr",{children:[g.jsx("th",{children:"Label"}),g.jsx("th",{children:"Probability"})]})}),g.jsx("tbody",{children:e.map(({label:a,confidence:r})=>g.jsxs("tr",{children:[g.jsx("td",{className:"capitalize",children:a}),g.jsx("td",{className:_e(r<=As&&"text-error"),children:r})]},a))})]})}):null}function wc({className:t,...e}){return g.jsxs("section",{className:_e("collapse bg-base-100 border-base-300 border collapse-arrow",t),...e,children:[g.jsx("input",{type:"checkbox"}),g.jsx("div",{className:"collapse-title font-semibold",children:"How to achieve precise predictions"}),g.jsxs("div",{className:"collapse-content",children:[g.jsx("h6",{children:"1️⃣ Number of objects per image"}),g.jsxs("ul",{children:[g.jsx("li",{children:"Single object per image → Best results."}),g.jsx("ul",{children:g.jsx("li",{children:"Inception V3 is optimized for recognizing one primary object."})}),g.jsx("li",{children:"Multiple objects → Accuracy drops."}),g.jsxs("ul",{children:[g.jsx("li",{children:"The model may predict the most prominent object, or mix predictions from different objects."}),g.jsx("li",{children:"If you want multiple object recognition, consider object detection models (e.g., Faster R-CNN, YOLO, SSD) instead."})]})]}),g.jsx("h6",{children:"2️⃣ Object size and placement"}),g.jsxs("ul",{children:[g.jsx("li",{children:"Centered, large object → More precise predictions."}),g.jsx("li",{children:"Small or off-center object → Model might miss it or focus on background."}),g.jsx("li",{children:"Cropping the object before feeding it improves precision."})]}),g.jsx("h6",{children:"3️⃣ Background"}),g.jsxs("ul",{children:[g.jsx("li",{children:"Simple, uncluttered backgrounds → Better predictions."}),g.jsx("li",{children:"Busy or similar-colored background → Model can confuse background with object features."}),g.jsx("li",{children:"Pretrained Inception V3 isn’t trained to ignore backgrounds, so segmentation or cropping helps."})]}),g.jsx("h6",{children:"4️⃣ Image quality"}),g.jsxs("ul",{children:[g.jsx("li",{children:"High resolution, sharp images → Higher accuracy."}),g.jsx("li",{children:"Blur, noise, compression artifacts → Reduces confidence and may misclassify."}),g.jsx("li",{children:"Consistent lighting helps because Inception V3 isn’t robust to extreme lighting changes."})]}),g.jsx("h6",{children:"5️⃣ Object viewpoint and orientation"}),g.jsxs("ul",{children:[g.jsx("li",{children:"Canonical views (like ImageNet standard: front, side, or typical pose) → Better."}),g.jsx("li",{children:"Extreme rotations, unusual angles, occlusions → Accuracy drops."})]}),g.jsx("h6",{children:"⚡ Summary (Photo-wise)"}),g.jsx("table",{className:"table",children:g.jsxs("tbody",{children:[g.jsxs("tr",{children:[g.jsx("th",{children:"Factor"}),g.jsx("th",{children:"Ideal for Inception V3"})]}),g.jsxs("tr",{children:[g.jsx("td",{children:"Objects per image"}),g.jsx("td",{children:"Single"})]}),g.jsxs("tr",{children:[g.jsx("td",{children:"Object placement"}),g.jsx("td",{children:"Centered"})]}),g.jsxs("tr",{children:[g.jsx("td",{children:"Object size"}),g.jsx("td",{children:"Large relative to frame"})]}),g.jsxs("tr",{children:[g.jsx("td",{children:"Background"}),g.jsx("td",{children:"Simple / clean"})]}),g.jsxs("tr",{children:[g.jsx("td",{children:"Image quality"}),g.jsx("td",{children:"Sharp, good lighting"})]}),g.jsxs("tr",{children:[g.jsx("td",{children:"Object viewpoint"}),g.jsx("td",{children:"Standard angles"})]})]})})]})]})}async function Tc(){await ci();try{return await $t(Gt)}catch{const t=await $t(fc,{fromTFHub:!0});return await t.save(Gt),t}}function Sc(){return mc({queryKey:["inception-v3-model"],queryFn:async()=>await Tc(),staleTime:1/0,gcTime:1/0})}function vc(){const[t,e]=xe.useState([]),[s,a]=xe.useState(!1),[r,n]=xe.useState(null),u=Cs(r),{isLoading:o,data:l,error:c}=Sc();return g.jsx("section",{className:_e("prose p-4",(o||!!c)&&"text-center"),children:o?g.jsxs(g.Fragment,{children:[g.jsx("p",{children:"Wait before model data will load..."}),g.jsx("p",{className:"loading loading-bars loading-xl"})]}):g.jsxs(g.Fragment,{children:[!!c&&g.jsxs(g.Fragment,{children:[g.jsx("h4",{children:"Oops, something went wrong"}),g.jsx("p",{className:"text-error",children:c.message}),g.jsx("p",{children:"Try to refresh the page"})]}),!c&&!!l&&g.jsxs(g.Fragment,{children:[g.jsx("h2",{children:"Inception V3 - neural network architecture for image classification"}),g.jsx("input",{type:"file",onChange:p=>{n(p.target.files?.[0]??null),a(!!p.target.files?.[0]),e([])},className:"file-input"}),s&&g.jsxs("div",{className:"text-center",children:[g.jsx("p",{children:"Wait before computation will finish..."}),g.jsx("p",{className:"loading loading-spinner loading-xl"})]}),g.jsx(Nc,{data:t}),u?g.jsx("img",{src:u,alt:"Loaded image",onLoad:p=>{bc({model:l,onSuccess:m=>{e(m),a(!1)},img:p.currentTarget})}}):g.jsx("p",{children:"Please select an image first"}),g.jsx(Ds,{}),g.jsx(wc,{})]})]})})}function Cc(){return g.jsx(dc,{children:g.jsx(vc,{})})}export{Cc as default};
