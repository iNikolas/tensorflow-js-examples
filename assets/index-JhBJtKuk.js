import{j as h,R as Te}from"./index-qOZqXyW5.js";import{c as de}from"./helpers-Cs5YJYh0.js";import{u as jt}from"./hooks-tg-nlY-M.js";import{M as Dt}from"./view-CsJ7CpAI.js";import{u as zt,Q as xt}from"./view-ql5VOHnI.js";import{o as T,b as v,c as g,d as ee,E as _,A as Lt,e as be,m as H,f as X,g as G,h as M,j as le,k as ke,B as Ct,l as Vt,D as Pt,n as Ft,L as Rt,p as k,q as he,S as Bt,M as qt,T as te,u as Z,v as We,w as Wt,x as re,R as Ht,y as Gt,z as Ut,C as Ne,F as Kt,G as nt,H as it,r as ot,I as ne,J as He,K as Ce,N as Ve,O as Pe,P as ut,Q as Jt,U as Xt,V as pt,W as lt,X as mt,Y as Qt,Z as q,_ as me,$ as ct,a0 as dt,a1 as Yt,a2 as Zt,a3 as Mt,a4 as ea,a5 as Q,a6 as ta,a7 as ht,a8 as aa,a9 as sa,aa as ra,ab as na,ac as ia,ad as oa,ae as Ge,af as Ue,ag as ua,ah as pa,ai as la,aj as ma,ak as ca,al as da,am as ha,an as yt,ao as Ke,ap as ya,aq as ae,ar as ft,as as fa,at as gt,au as ga,av as ba,aw as Na,ax as wa,ay as bt,az as Ta,aA as Sa,aB as va,aC as ka,aD as Oa,aE as _a,aF as Aa,aG as Ia,aH as Ea,aI as $a,aJ as ja,aK as Da,aL as za,aM as xa,aN as La,aO as Nt,aP as x,aQ as Ca,aR as Va,aS as Pa,aT as Fa,aU as Ra,aV as Ba,aW as qa,aX as Wa,aY as Ha,aZ as Ga,a_ as Ua,a$ as Ka,b0 as Ja,b1 as Xa,b2 as Qa,b3 as Ya,b4 as Za,b5 as Ma,b6 as es,b7 as ts,b8 as as,b9 as ss,ba as rs,bb as ns,bc as is,bd as os,be as us,bf as ps,bg as ls,bh as ms,bi as cs,bj as ds,bk as hs,bl as ys,bm as fs,bn as gs,bo as bs,bp as Ns,bq as ws,br as Ts,bs as Ss,bt as vs,bu as ks,bv as Os,bw as _s,bx as As,by as Is,bz as Es,bA as $s,bB as js,bC as Ds,bD as zs,bE as xs,bF as Ls,bG as Cs,bH as Vs,bI as Ps,bJ as Fs,bK as Rs,bL as Bs,bM as qs,bN as Ws,bO as Hs,bP as Gs,i as wt,bQ as Us,bR as Ks,bS as Js,bT as Xs,bU as Qs,bV as Ys,bW as Zs,bX as Ms,bY as er,bZ as tr,b_ as ar,b$ as sr,c0 as rr,c1 as nr,c2 as ir,c3 as or,c4 as ur,c5 as pr,c6 as lr,c7 as mr,c8 as cr,c9 as dr,ca as hr,cb as yr,cc as fr,cd as gr,ce as br,cf as Nr,cg as wr,ch as Tr,ci as Sr,cj as vr,ck as kr,cl as Or,cm as _r,cn as Ar,co as Ir,cp as Er,cq as $r,cr as jr,cs as Dr,ct as zr,cu as xr,cv as Lr,cw as Cr,cx as Vr,cy as Pr,cz as Fr,cA as Rr,cB as Br,cC as qr,cD as Wr,cE as Hr,cF as Gr,cG as Ur,cH as Kr,cI as Jr,cJ as Xr,cK as Qr,cL as Yr,cM as Zr,s as Mr,cN as en,cO as tn,cP as an,a as se,cQ as sn,cR as rn,cS as nn,cT as on,cU as un,cV as pn,cW as ln,cX as mn,cY as Tt,cZ as cn,c_ as dn,c$ as hn,d0 as yn,d1 as ie,d2 as fn,d3 as gn,d4 as bn,d5 as Nn,d6 as B,t as V,d7 as ye,d8 as wn,d9 as Tn}from"./register_all_kernels-YBgeH-G1.js";import{a as Sn,f as vn}from"./browser-B_FScVPB.js";/**
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
 */function kn(a){v(Array.isArray(a),()=>"The argument passed to tf.addN() must be a list of tensors"),v(a.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${a.length}`);const e=a.map((r,i)=>g(r,`tensors${i}`,"addN")),t=e[0];e.forEach(r=>{if(r.dtype!==t.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),e.forEach(r=>{if(!ee(r.shape,t.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const s=e;return _.runKernel(Lt,s)}const On=T({addN_:kn});/**
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
 */function _n(a,e,t,s,r,i){const u=g(a,"forgetBias","basicLSTMCell"),o=g(e,"lstmKernel","basicLSTMCell"),p=g(t,"lstmBias","basicLSTMCell"),l=g(s,"data","basicLSTMCell"),m=g(r,"c","basicLSTMCell"),c=g(i,"h","basicLSTMCell"),d=be([l,c],1),y=H(d,o),w=X(y,p),b=w.shape[0],f=w.shape[1]/4,N=[b,f],O=G(w,[0,0],N),$=G(w,[0,f],N),S=G(w,[0,f*2],N),E=G(w,[0,f*3],N),D=X(M(le(O),ke($)),M(m,le(X(u,S)))),z=M(ke(D),le(E));return[D,z]}const An=T({basicLSTMCell_:_n});/**
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
 */function In(a,e){const t=g(a,"x","bitwiseAnd"),s=g(e,"y","bitwiseAnd");if(!ee(t.shape,s.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${t.shape}, y: ${s.shape}`);if(t.dtype!=="int32"||s.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${t.dtype} and type of y: ${s.dtype}`);const r={a:t,b:s};return _.runKernel(Ct,r)}const En=T({bitwiseAnd_:In});/**
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
 */function $n(a,e){const t=g(a,"s0","broadcastArgs","int32"),s=g(e,"s1","broadcastArgs","int32");if(t.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${t.rank}`);if(s.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${s.rank}`);const r={s0:t,s1:s};return _.runKernel(Vt,r)}const jn=T({broadcastArgs_:$n});/**
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
 */function Dn(a){const t={x:g(a,"x","diag")};return _.runKernel(Pt,t)}const zn=T({diag_:Dn});/**
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
 */function xn(a,e){const t=g(a,"x","ensureShape","string_or_numeric");if(!Ft(t.shape,e))throw new Error(`EnsureShape: Shape of tensor ${t.shape} is not compatible with expected shape ${e}`);return a}const Ln=T({ensureShape_:xn});/**
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
 */function Cn(a,e,t){if(t<=0)throw new Error("The number of values should be positive.");const s={start:a,stop:e,num:t};return _.runKernel(Rt,{},s)}/**
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
 */const oe=2147483648;function Vn(a,e,t="left"){const s=g(a,"sortedSequence","searchSorted"),r=g(e,"values","searchSorted"),i=s.shape[s.shape.length-1],u=r.shape[r.shape.length-1],o=k(s,[-1,i]),p=k(r,[-1,u]);if(o.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(o.shape[0]!==p.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(he(p.shape)>=oe)throw new Error(`values tensor size must less than ${oe}`);if(o.shape[1]>=oe)throw new Error(`trailing dim_size must less than ${oe} for int32 output type, was ${o.shape[1]}`);const l={sortedSequence:o,values:p},m={side:t};return _.runKernel(Bt,l,m)}const Fe=T({searchSorted_:Vn});/**
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
 */function Pn(a,e){return Fe(a,e,"left")}/**
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
 */function Fn(a,e,t,s,r=!1){const u={x:g(a,"x","maxPoolWithArgmax")},o={filterSize:e,strides:t,pad:s,includeBatchInIndex:r},p=_.runKernel(qt,u,o);return{result:p[0],indexes:p[1]}}const Rn=T({maxPoolWithArgmax_:Fn});/**
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
 */function Bn(a,e,{indexing:t="xy"}={}){if(t!=="xy"&&t!=="ij")throw new TypeError(`${t} is not a valid third argument to meshgrid`);if(a===void 0)return[];let s=g(a,"x","meshgrid",a instanceof te?a.dtype:"float32");if(e===void 0)return[s];let r=g(e,"y","meshgrid",e instanceof te?e.dtype:"float32");const i=he(s.shape),u=he(r.shape);return t==="xy"?(s=k(s,[1,-1]),r=k(r,[-1,1]),[H(Z([u,1],s.dtype),s),H(r,Z([1,i],r.dtype))]):(s=k(s,[-1,1]),r=k(r,[1,-1]),[H(s,Z([1,u],s.dtype)),H(Z([i,1],r.dtype),r)])}function qn(a,e,t,s){const r=g(e,"data","multiRNNCell"),i=We(t,"c","multiRNNCell"),u=We(s,"h","multiRNNCell");let o=r;const p=[];for(let c=0;c<a.length;c++){const d=a[c](o,i[c],u[c]);p.push(d[0]),p.push(d[1]),o=d[1]}const l=[],m=[];for(let c=0;c<p.length;c+=2)l.push(p[c]),m.push(p[c+1]);return[l,m]}const Wn=T({multiRNNCell_:qn});/**
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
 */function Hn(a,e,t,s=!1){const r=g(a,"logits","multinomial"),i=r.size,u=r.rank;if(i<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${i}.`);if(u>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${u}`);t=t||Math.random();const p={logits:u===1?k(r,[1,-1]):r},l={numSamples:e,seed:t,normalized:s},m=_.runKernel(Wt,p,l);return u===1?k(m,[m.size]):m}const Gn=T({multinomial_:Hn});function Un(a,e){const t=g(a,"v1","outerProduct"),s=g(e,"v2","outerProduct");v(t.rank===1&&s.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${t.rank} and ${s.rank}.`);const r=k(t,[-1,1]),i=k(s,[1,-1]);return H(r,i)}const Kn=T({outerProduct_:Un});function Jn(a,e,t=0){return v(e.length===2,()=>"Invalid number of paddings. Must be length of 2."),re(a,[e],t)}const Xn=T({pad1d_:Jn});function Qn(a,e,t=0){return v(e.length===2&&e[0].length===2&&e[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),re(a,e,t)}const Yn=T({pad2d_:Qn});function Zn(a,e,t=0){return v(e.length===3&&e[0].length===2&&e[1].length===2&&e[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),re(a,e,t)}const Mn=T({pad3d_:Zn});function ei(a,e,t=0){return v(e.length===4&&e[0].length===2&&e[1].length===2&&e[2].length===2&&e[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),re(a,e,t)}const ti=T({pad4d_:ei});/**
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
 */function ai(a,e,t,s){const r=a.map((m,c)=>g(m,`tensors${c}`,"raggedGather","int32")),i=g(e,"paramsDenseValues","raggedGather"),u=g(t,"indices","raggedGather","int32"),o={paramsNestedSplits:r,paramsDenseValues:i,indices:u},p={outputRaggedRank:s},l=_.runKernel(Ht,o,p);return{outputNestedSplits:l.slice(0,l.length-1),outputDenseValues:l[l.length-1]}}const si=T({raggedGather_:ai});/**
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
 */function ri(a,e,t){const s=g(a,"starts","raggedRange"),r=g(e,"limits","raggedRange",s.dtype),i=g(t,"deltas","raggedRange",s.dtype),u={starts:s,limits:r,deltas:i},o=_.runKernel(Gt,u);return{rtNestedSplits:o[0],rtDenseValues:o[1]}}const ni=T({raggedRange_:ri});/**
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
 */function ii(a,e,t,s,r){const i=g(a,"shape","raggedTensorToTensor","int32"),u=g(e,"values","raggedTensorToTensor"),o=g(t,"defaultValue","raggedTensorToTensor",u.dtype),p=s.map((c,d)=>g(c,`tensors${d}`,"raggedTensorToTensor","int32")),l={shape:i,values:u,defaultValue:o,rowPartitionTensors:p},m={rowPartitionTypes:r};return _.runKernel(Ut,l,m)}const oi=T({raggedTensorToTensor_:ii});/**
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
 */function ui(a,e,t){Ne(a);const s=he(a);let r=null;if(t==null||t==="float32")r=new Float32Array(s);else if(t==="int32")r=new Int32Array(s);else if(t==="bool")r=new Uint8Array(s);else throw new Error(`Unknown data type ${t}`);for(let i=0;i<s;i++)r[i]=e();return _.makeTensor(r,a,t)}const pi=T({rand_:ui});/**
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
 */function li(a,e,t=1,s="float32",r){if(Ne(a),t==null&&(t=1),s==null&&(s="float32"),s!=="float32"&&s!=="int32")throw new Error(`Unsupported data type ${s}`);const i=new Kt(e,t,s,r),u=nt(a,s);for(let o=0;o<u.values.length;o++)u.values[o]=i.nextValue();return u.toTensor()}const mi=T({randomGamma_:li});/**
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
 */function ci(a,e,t){if(e!=null&&e==="bool")throw new Error(`Unsupported data type ${e}`);return it(a,0,1,e,t)}const di=T({randomStandardNormal_:ci});/**
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
 */function hi(a,e,t,s){return ot(a,e,t,"int32",s)}const yi=T({randomUniformInt_:hi});/**
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
 */function fi(a){const e=g(a,"x","reverse");return v(e.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${e.rank}.`),ne(e,0)}const gi=T({reverse1d_:fi});/**
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
 */function bi(a,e){const t=g(a,"x","reverse");return v(t.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${t.rank}.`),ne(t,e)}const Ni=T({reverse2d_:bi});/**
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
 */function wi(a,e){const t=g(a,"x","reverse");return v(t.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${t.rank}.`),ne(t,e)}const Ti=T({reverse3d_:wi});/**
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
 */function Si(a,e){const t=g(a,"x","reverse");return v(t.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${t.rank}.`),ne(t,e)}const vi=T({reverse4d_:Si});/**
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
 */async function ki(a,e){const t=g(a,"x","setdiff1d"),s=g(e,"y","setdiff1d");v(t.dtype===s.dtype,()=>`x and y should have the same dtype, but got x (${t.dtype}) and y (${s.dtype}).`),v(t.rank===1,()=>`x should be 1D tensor, but got x (${t.shape}).`),v(s.rank===1,()=>`y should be 1D tensor, but got y (${s.shape}).`);const r=await t.data(),i=await s.data(),u=new Set(i);let o=0;for(let m=0;m<r.length;m++)u.has(r[m])||o++;const p=new He([o],t.dtype),l=new He([o],"int32");for(let m=0,c=0;m<r.length;m++)u.has(r[m])||(p.values[c]=r[m],l.values[c]=m,c++);return[p.toTensor(),l.toTensor()]}const Oi=ki;/**
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
 */function _i(a,e,t){if(Ce(a),e!=null&&e.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const s=Ve(a,t);if(s.length!==4&&s.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return Pe(a,e,s,t)}/**
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
 */function Ai(a,e,t){if(Ce(a),e!=null&&e.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const s=Ve(a,t);if(s.length!==5&&s.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return Pe(a,e,s,t)}/**
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
 */function Ii(a,e,t){if(Ce(a),e!=null&&e.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const s=Ve(a,t);if(s.length!==6&&s.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return e=e||s,Pe(a,e,s,t)}/**
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
 */function Ei(a,e,t){const s=g(a,"tensor","tensorScatterupdate"),r=g(e,"indices","tensorScatterupdate","int32"),i=g(t,"updates","tensorScatterupdate");if(ut(i,r,s.shape),s.dtype!==i.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${s.dtype} and ${i.dtype}.`);const u={tensor:s,indices:r,updates:i},o={};return _.runKernel(Jt,u,o)}const $i=T({tensorScatterUpdate_:Ei});/**
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
 */function ji(a,e){return Fe(a,e,"right")}/**
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
 */async function Di(a){const e=g(a,"condition","whereAsync","bool"),t=await e.data(),s=Xt(e.shape,t);return a!==e&&e.dispose(),s}const St=Di;/**
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
 */async function zi(a,e,t){const s=g(a,"tensor","boolMask"),r=g(e,"mask","boolMask","bool"),i=t??0,u=r.rank,o=s.shape;v(u>0,()=>"mask cannot be scalar"),pt(o.slice(i,i+u),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let p=1;for(let b=i;b<i+u;b++)p*=o[b];const l=o.slice(0,i).concat([p],o.slice(i+u)),m=k(s,l),c=k(r,[-1]),d=await St(c),y=lt(d,[1]),w=mt(m,y,i);return a!==s&&s.dispose(),e!==r&&r.dispose(),y.dispose(),m.dispose(),c.dispose(),d.dispose(),w}const xi=zi;/**
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
 */function Li(a,e,t,s,r=!0){const i=g(a,"v","movingAverage"),u=g(e,"x","movingAverage"),o=g(t,"decay","movingAverage");Qt(i,u),v(ee(i.shape,u.shape),()=>"Shape mismatch in v and x");const p=q(1),l=me(p,o);let m=M(me(u,i),l);if(r){v(s!=null,()=>"When using zeroDebias: true, step is required.");const c=g(s,"step","movingAverage");m=ct(m,me(p,dt(o,c)))}return X(i,m)}const Ci=T({movingAverage_:Li});/**
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
 */function Vi(a,e,t){Ne(t);const s=g(a,"indices","scatterND","int32"),r=g(e,"updates","scatterND");ut(r,s,t);const i={indices:s,updates:r},u={shape:t};return _.runKernel(Yt,i,u)}const Pi=T({scatterND_:Vi});function Fi(a,e,t,s){if(a.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${a.dtype}.`);if(a.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${a.shape}.`);const r=a.rank>0?a.shape[0]:1,i=a.rank>1?a.shape[1]:1;if(t.length!==i)throw new Error(`outputShape has incorrect number of elements:, ${t.length}, should be: ${i}.`);const u=e.size;if(!(e.rank===0||e.rank===1&&u===r))throw new Error(`sparseValues has incorrect shape ${e.shape}, should be [] or [${r}]`);if(e.dtype!==s.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
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
 */function Ri(a,e,t,s=0){Ne(t);const r=g(a,"sparseIndices","sparseToDense","int32"),i=g(e,"sparseValues","sparseToDense","string_or_numeric"),u=g(s,"defaultValue","sparseToDense",i.dtype);Fi(r,i,t,u);const o={sparseIndices:r,sparseValues:i,defaultValue:u},p={outputShape:t};return _.runKernel(Zt,o,p)}const Bi=T({sparseToDense_:Ri});/**
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
 */function qi(a,e){const t=g(e,"indices","gatherND","int32"),r={params:g(a,"x","gatherND","string_or_numeric"),indices:t};return _.runKernel(Mt,r)}const Wi=T({gatherND_:qi});/**
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
 */async function Hi(a,e,t=1){const s=g(a,"predictions","inTopK"),r=g(e,"targets","inTopK");v(s.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${s.rank}`),v(s.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${s.rank} and targets rank ${r.rank}`),pt(s.shape.slice(0,s.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const i=s.shape[s.shape.length-1];v(t>0&&t<=i,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${i}), but got ${t}`);const u=await s.data(),o=await r.data(),[p,l]=[u.length/i,i],m=ea("bool",p);for(let c=0;c<p;c++){const d=c*l,y=u.subarray(d,d+l),w=[];for(let b=0;b<y.length;b++)w.push({value:y[b],index:b});w.sort((b,f)=>f.value-b.value),m[c]=0;for(let b=0;b<t;b++)if(w[b].index===o[c]){m[c]=1;break}}return a!==s&&s.dispose(),e!==r&&r.dispose(),Q(m,r.shape,"bool")}const Gi=Hi;/**
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
 */function Ui({x:a,filter:e,strides:t,pad:s,dataFormat:r="NHWC",dilations:i=[1,1],dimRoundingMode:u,bias:o,activation:p="linear",preluActivationWeights:l,leakyreluAlpha:m}){if(ta(_.state.gradientDepth,p)===!1){let E=ht(a,e,t,s,r,i,u);return o!=null&&(E=X(E,o)),aa(E,p,l,m)}const c=g(a,"x","depthwiseConv2d","float32"),d=g(e,"filter","depthwiseConv2d","float32");let y=c,w=!1;c.rank===3&&(w=!0,y=k(c,[1,c.shape[0],c.shape[1],c.shape[2]])),v(y.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${y.rank}.`),v(d.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${d.rank}.`),v(y.shape[3]===d.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${y.shape[3]}) must match the inChannels dimension in filter ${d.shape[2]}.`),i==null&&(i=[1,1]),v(sa(t,i),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`),ra("fused depthwiseConv2d",s,u);const b=na(y.shape,d.shape,t,i,s,u,!0);let f;o!=null&&(f=g(o,"bias","fused conv2d"),[f]=ia(f,c),oa(b.outShape,f.shape));let N;l!=null&&(N=g(l,"prelu weights","fused depthwiseConv2d"));const O=(E,D)=>{v(ua(i),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${i}'`);const[z,J,C,P]=D,we=pa(E,C,p),Be=la(J.shape,we,z,t,s,i,u),qe=ma(J,we,z.shape,t,s,i,u);if(P!=null){const $t=ca(f,we);return[Be,qe,$t]}return[Be,qe]},$={x:y,filter:d,bias:f,preluActivationWeights:N},S={strides:t,pad:s,dataFormat:r,dilations:i,dimRoundingMode:u,activation:p,leakyreluAlpha:m};return o==null?Ge((D,z,J)=>{let C=_.runKernel(Ue,$,S);return J([z,D,C]),w&&(C=k(C,[C.shape[1],C.shape[2],C.shape[3]])),{value:C,gradFunc:O}})(y,d):Ge((D,z,J,C)=>{let P=_.runKernel(Ue,$,S);return C([z,D,P,J]),w&&(P=k(P,[P.shape[1],P.shape[2],P.shape[3]])),{value:P,gradFunc:O}})(y,d,f)}const Ki=T({fusedDepthwiseConv2d_:Ui});/**
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
 */const Ji=Object.freeze(Object.defineProperty({__proto__:null,conv2d:da,depthwiseConv2d:Ki,matMul:ha},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Xi="model",Qi=".json",Yi=".weights.bin";function Je(a){return new Promise(e=>setTimeout(e)).then(a)}class U{constructor(e){if(!ae().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(U.URL_SCHEME)&&(e=e.slice(U.URL_SCHEME.length)),(e==null||e.length===0)&&(e=Xi),this.modelJsonFileName=e+Qi,this.weightDataFileName=e+Yi}async save(e){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const t=ft.join(e.weightData),s=window.URL.createObjectURL(new Blob([t],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const r=[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}],i=fa(e,r),u=window.URL.createObjectURL(new Blob([JSON.stringify(i)],{type:"application/json"})),o=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(o.download=this.modelJsonFileName,o.href=u,await Je(()=>o.dispatchEvent(new MouseEvent("click"))),e.weightData!=null){const p=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;p.download=this.weightDataFileName,p.href=s,await Je(()=>p.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:gt(e)}}}}U.URL_SCHEME="downloads://";class Zi{constructor(e){if(e==null||e.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{const s=new FileReader;s.onload=r=>{const i=JSON.parse(r.target.result),u=i.modelTopology;if(u==null){t(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(i.weightsManifest==null){t(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){e({modelTopology:u});return}const p=yt(i,l=>this.loadWeights(l));e(p)},s.onerror=r=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(e){const t=[],s=[];for(const u of e)t.push(...u.weights),s.push(...u.paths);const r=this.checkManifestAndWeightFiles(e),i=s.map(u=>this.loadWeightsFile(u,r[u]));return Promise.all(i).then(u=>[t,u])}loadWeightsFile(e,t){return new Promise((s,r)=>{const i=new FileReader;i.onload=u=>{const o=u.target.result;s(o)},i.onerror=u=>r(`Failed to weights data from file of path '${e}'.`),i.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){const t=[],s=this.weightsFiles.map(i=>Ke(i.name)),r={};for(const i of e)i.paths.forEach(u=>{const o=Ke(u);if(t.indexOf(o)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${o}'`);if(t.push(o),s.indexOf(o)===-1)throw new Error(`Weight file with basename '${o}' is not provided.`);r[u]=this.weightsFiles[s.indexOf(o)]});if(t.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const Mi=a=>ae().getBool("IS_BROWSER")&&!Array.isArray(a)&&a.startsWith(U.URL_SCHEME)?eo(a.slice(U.URL_SCHEME.length)):null;ya.registerSaveRouter(Mi);function eo(a="model"){return new U(a)}function to(a){return new Zi(a)}/**
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
 */class Se{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class vt{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class ao{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}}function so(a,e,t,s){const r=arguments;return new ao(kt(...r))}function kt(a,e,t,s){return arguments.length===1?a.modelTopology!=null||a.weightSpecs!=null?new Se(a):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Se({modelTopology:a})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Se({modelTopology:a,weightSpecs:e,weightData:t,trainingConfig:s}))}function ro(a){return new vt(a)}function no(a){return new vt(a)}/**
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
 */const Ot=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:ft,browserFiles:to,browserHTTPRequest:ga,concatenateArrayBuffers:ba,copyModel:Na,decodeWeights:wa,decodeWeightsStream:bt,encodeWeights:Ta,fromMemory:so,fromMemorySync:kt,getLoadHandlers:Sa,getModelArtifactsForJSON:yt,getModelArtifactsForJSONSync:va,getModelArtifactsInfoForJSON:gt,getSaveHandlers:ka,getWeightSpecs:Oa,http:_a,isHTTPScheme:Aa,listModels:Ia,loadWeights:Ea,moveModel:$a,registerLoadRouter:ja,registerSaveRouter:Da,removeModel:za,weightsLoaderFactory:xa,withSaveHandler:ro,withSaveHandlerSync:no},Symbol.toStringTag,{value:"Module"}));/**
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
 */const io={};function _t(a){return io[a]}/**
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
 */function n(a,e,t,s,r){const i=e.inputParams[a];if(i&&i.inputIndexStart!==void 0){const o=i.inputIndexStart,p=i.inputIndexEnd===0?void 0:i.inputIndexEnd===void 0?o+1:i.inputIndexEnd,l=o<0?e.inputNames.length+o:o;if(i.type==="tensor")return A(e.inputNames[l],t,s,r);if(i.type==="tensors"){const d=e.inputs.slice(o,p);return e.inputNames.slice(o,p).filter((w,b)=>{var f;return((f=d[b])===null||f===void 0?void 0:f.op)!=="NoOp"}).map(w=>A(w,t,s,r))}const m=A(e.inputNames[l],t,s,r),c=m.dataSync();return i.type==="number"?c[0]:La(m.shape,c)}const u=e.attrParams[a];return u&&u.value}function A(a,e,t,s){const[r,i]=j(a,t);if(s!=null){const o=s.getHashTableHandleByName(r);if(o!=null)return o}const u=t.currentContextIds.find(o=>!!e[fe(r,o)]);return u!==void 0?e[fe(r,u)][i]:void 0}function Xe(a,e,t){return e[fe(a,t.currentContextId)]}function F(a,e){const[t,s,r]=j(a,e);return[fe(t,e&&e.currentContextId),s,r]}function fe(a,e){return e?`${a}-${e}`:a}function j(a,e){if(a==="")return["",0,void 0];const t=e!=null&&e.parseNodeNameCache!=null;if(t){const i=e.parseNodeNameCache.get(a);if(i!=null)return i}const s=a.split(":");let r;if(s.length===1)r=[a,0,void 0];else{const i=s[0],u=s.length===3?s[1]:void 0,o=Number(s[s.length-1]);r=[i,o,u]}return t&&e.parseNodeNameCache.set(a,r),r}function ce(a,e,t){let s=n("pad",a,e,t);if(s==="explicit"){s=n("explicitPaddings",a,e,t);const r=[[0,0],[0,0],[0,0],[0,0]];for(let i=0;i<4;i++)r[i][0]=s[i*2],r[i][1]=s[i*2+1];return r}return s}function R(a){return a.kept?a:Nt(a)}/**
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
 */const oo=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],uo=Object.freeze(Object.defineProperty({__proto__:null,json:oo},Symbol.toStringTag,{value:"Module"}));/**
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
 */const po=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],lo=Object.freeze(Object.defineProperty({__proto__:null,json:po},Symbol.toStringTag,{value:"Module"}));/**
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
 */const mo=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],co=Object.freeze(Object.defineProperty({__proto__:null,json:mo},Symbol.toStringTag,{value:"Module"}));/**
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
 */const ho=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],yo=Object.freeze(Object.defineProperty({__proto__:null,json:ho},Symbol.toStringTag,{value:"Module"}));/**
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
 */const fo=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniformInt",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number"},{tfName:"maxval",name:"maxval",type:"number"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],go=Object.freeze(Object.defineProperty({__proto__:null,json:fo},Symbol.toStringTag,{value:"Module"}));/**
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
 */const bo=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],No=Object.freeze(Object.defineProperty({__proto__:null,json:bo},Symbol.toStringTag,{value:"Module"}));/**
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
 */const wo=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],To=Object.freeze(Object.defineProperty({__proto__:null,json:wo},Symbol.toStringTag,{value:"Module"}));/**
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
 */const So=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],vo=Object.freeze(Object.defineProperty({__proto__:null,json:So},Symbol.toStringTag,{value:"Module"}));/**
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
 */const ko=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],Oo=Object.freeze(Object.defineProperty({__proto__:null,json:ko},Symbol.toStringTag,{value:"Module"}));/**
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
 */const _o=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],Ao=Object.freeze(Object.defineProperty({__proto__:null,json:_o},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Io=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BitwiseAnd",category:"logical",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}]}],Eo=Object.freeze(Object.defineProperty({__proto__:null,json:Io},Symbol.toStringTag,{value:"Module"}));/**
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
 */const $o=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}],jo=Object.freeze(Object.defineProperty({__proto__:null,json:$o},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Do=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]}],zo=Object.freeze(Object.defineProperty({__proto__:null,json:Do},Symbol.toStringTag,{value:"Module"}));/**
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
 */const xo=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],Lo=Object.freeze(Object.defineProperty({__proto__:null,json:xo},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Co=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],Vo=Object.freeze(Object.defineProperty({__proto__:null,json:Co},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Po=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],Fo=Object.freeze(Object.defineProperty({__proto__:null,json:Po},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Ro=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],Bo=Object.freeze(Object.defineProperty({__proto__:null,json:Ro},Symbol.toStringTag,{value:"Module"}));/**
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
 */const qo=[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],Wo=Object.freeze(Object.defineProperty({__proto__:null,json:qo},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Ho=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"EnsureShape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],Go=Object.freeze(Object.defineProperty({__proto__:null,json:Ho},Symbol.toStringTag,{value:"Module"}));/**
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
 */class Qe{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const e=[uo,lo,co,yo,go,No,To,vo,Oo,Ao,Eo,jo,zo,Lo,Vo,Fo,Bo,Wo,Go],t=[].concat(...e.map(s=>s.json));this.opMappers=t.reduce((s,r)=>(s[r.tfOpName]=r,s),{})}transformGraph(e,t={}){const s=e.node,r=[],i=[],u=[],o=s.reduce((b,f)=>(b[f.name]=this.mapNode(f),f.op.startsWith("Placeholder")?r.push(b[f.name]):f.op==="Const"?i.push(b[f.name]):(f.input==null||f.input.length===0)&&u.push(b[f.name]),b),{});let p=[];const l=[];let m={},c={};t!=null&&(m=this.mapSignatureEntries(t.inputs),c=this.mapSignatureEntries(t.outputs));const d=Object.keys(o);d.forEach(b=>{const f=o[b];f.inputNames.forEach((N,O)=>{const[$,,S]=F(N),E=o[$];if(E.outputs!=null){const D=E.outputs.indexOf(S);if(D!==-1){const z=`${$}:${D}`;f.inputNames[O]=z}}f.inputs.push(E),E.children.push(f)})}),Object.keys(c).length===0?d.forEach(b=>{const f=o[b];f.children.length===0&&l.push(f)}):Object.keys(c).forEach(b=>{const[f]=F(b),N=o[f];N!=null&&(N.signatureKey=c[b],l.push(N))}),Object.keys(m).length>0?Object.keys(m).forEach(b=>{const[f]=F(b),N=o[f];N&&(N.signatureKey=m[b],p.push(N))}):p=r;let y={};e.library!=null&&e.library.function!=null&&(y=e.library.function.reduce((b,f)=>(b[f.signature.name]=this.mapFunction(f),b),{}));const w={nodes:o,inputs:p,outputs:l,weights:i,placeholders:r,signature:t,functions:y};return u.length>0&&(w.initNodes=u),w}mapSignatureEntries(e){return Object.keys(e||{}).reduce((t,s)=>(t[e[s].name]=s,t),{})}mapNode(e){const t=_t(e.op)||this.opMappers[e.op]||{};e.attr==null&&(e.attr={});const s={name:e.name,op:e.op,category:t.category,inputNames:(e.input||[]).map(r=>r.startsWith("^")?r.slice(1):r),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:t.outputs};return t.inputs!=null&&(s.inputParams=t.inputs.reduce((r,i)=>(r[i.name]={type:i.type,inputIndexStart:i.start,inputIndexEnd:i.end},r),{})),t.attrs!=null&&(s.attrParams=t.attrs.reduce((r,i)=>{const u=i.type;let o;switch(i.type){case"string":o=Oe(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Oe(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"string[]":o=De(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=De(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"number":o=Ae(e.attr,i.tfName,i.defaultValue||0),o===void 0&&i.tfDeprecatedName&&(o=Ae(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"number[]":o=je(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=je(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"bool":o=_e(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=_e(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"bool[]":o=xe(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=xe(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"shape":o=$e(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=$e(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"shape[]":o=ze(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=ze(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"dtype":o=Ie(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Ie(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"dtype[]":o=Ee(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Ee(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"func":o=Ye(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Ye(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${i.type} for op: ${e.op}`)}return r[i.name]={value:o,type:u},r},{})),s}mapFunction(e){const t=e.nodeDef,s=[],r=[];let i={};t!=null&&(i=t.reduce((c,d)=>(c[d.name]=this.mapNode(d),d.op==="Const"&&r.push(c[d.name]),c),{}));const u=[],o=[];e.signature.inputArg.forEach(c=>{const[d]=F(c.name),y={name:d,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:Re(c.type),type:"dtype"}},children:[]};y.signatureKey=c.name,u.push(y),i[d]=y}),Object.keys(i).forEach(c=>{const d=i[c];d.inputNames.forEach((y,w)=>{const[b,,f]=F(y),N=i[b];if(N.outputs!=null){const O=N.outputs.indexOf(f);if(O!==-1){const $=`${b}:${O}`;d.inputNames[w]=$}}d.inputs.push(N),N.children.push(d)})});const l=e.ret;e.signature.outputArg.forEach(c=>{const[d,y]=F(l[c.name]),w=i[d];w!=null&&(w.defaultOutput=y,o.push(w))});const m=this.mapArgsToSignature(e);return{nodes:i,inputs:u,outputs:o,weights:r,placeholders:s,signature:m}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((t,s)=>(t[s.name]=this.mapArgToTensorInfo(s),t),{}),outputs:e.signature.outputArg.reduce((t,s)=>(t[s.name]=this.mapArgToTensorInfo(s,e.ret),t),{})}}mapArgToTensorInfo(e,t){let s=e.name;return t!=null&&(s=t[s]),{name:s,dtype:e.type}}}function Uo(a){const e=ae().global;if(typeof e.atob<"u")return e.atob(a);if(typeof Buffer<"u")return new Buffer(a,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function At(a,e){const t=Array.isArray(a)?String.fromCharCode.apply(null,a):Uo(a);return e?t:t.toLowerCase()}function Oe(a,e,t,s=!1){const r=a[e];return r!=null?At(r.s,s):t}function _e(a,e,t){const s=a[e];return s?s.b:t}function Ae(a,e,t){const s=a[e]||{},r=s.i!=null?s.i:s.f!=null?s.f:t;return typeof r=="number"?r:parseInt(r,10)}function Re(a){switch(typeof a=="string"&&(a=x[a]),a){case x.DT_FLOAT:case x.DT_HALF:return"float32";case x.DT_INT32:case x.DT_INT64:case x.DT_INT8:case x.DT_UINT8:return"int32";case x.DT_BOOL:return"bool";case x.DT_DOUBLE:return"float32";case x.DT_STRING:return"string";case x.DT_COMPLEX64:case x.DT_COMPLEX128:return"complex64";default:return null}}function Ye(a,e,t){const s=a[e];return s&&s.func?s.func.name:t}function Ie(a,e,t){const s=a[e];return s&&s.type?Re(s.type):t}function Ee(a,e,t){const s=a[e];return s&&s.list&&s.list.type?s.list.type.map(r=>Re(r)):t}function It(a){if(!a.unknownRank)return a.dim!=null?a.dim.map(e=>typeof e.size=="number"?e.size:parseInt(e.size,10)):[]}function $e(a,e,t){const s=a[e];return s&&s.shape?It(s.shape):t}function je(a,e,t){const s=a[e];return s?((s.list.f&&s.list.f.length?s.list.f:s.list.i)||[]).map(r=>typeof r=="number"?r:parseInt(r,10)):t}function De(a,e,t,s=!1){const r=a[e];return r&&r.list&&r.list.s?r.list.s.map(i=>At(i,s)):t}function ze(a,e,t){const s=a[e];return s&&s.list&&s.list.shape?s.list.shape.map(r=>It(r)):t}function xe(a,e,t){const s=a[e];return s&&s.list&&s.list.b?s.list.b:t}/**
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
 */class Ko{constructor(e,t,s){this.node=e,this.tensorMap=t,this.context=s,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(r=>this.getInput(r)),e.rawAttrs!=null&&(this.attrs=Object.keys(e.rawAttrs).reduce((r,i)=>(r[i]=this.getAttr(i),r),{}))}getInput(e){return A(e,this.tensorMap,this.context)}getAttr(e,t){const s=this.node.rawAttrs[e];if(s.tensor!=null)return A(e,this.tensorMap,this.context);if(s.i!=null||s.f!=null)return Ae(this.node.rawAttrs,e,t);if(s.s!=null)return Oe(this.node.rawAttrs,e,t);if(s.b!=null)return _e(this.node.rawAttrs,e,t);if(s.shape!=null)return $e(this.node.rawAttrs,e,t);if(s.type!=null)return Ie(this.node.rawAttrs,e,t);if(s.list!=null){if(s.list.i!=null||s.list.f!=null)return je(this.node.rawAttrs,e,t);if(s.list.s!=null)return De(this.node.rawAttrs,e,t);if(s.list.shape!=null)return ze(this.node.rawAttrs,e,t);if(s.list.b!=null)return xe(this.node.rawAttrs,e,t);if(s.list.type!=null)return Ee(this.node.rawAttrs,e,t)}return t}}/**
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
 */const I=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:Ca,abs:Va,acos:Pa,acosh:Fa,add:X,addN:On,all:Ra,any:Ba,argMax:qa,argMin:Wa,asin:Ha,asinh:Ga,atan:Ua,atan2:Ka,atanh:Ja,avgPool:Xa,avgPool3d:Qa,basicLSTMCell:An,batchNorm:Ya,batchNorm2d:Za,batchNorm3d:Ma,batchNorm4d:es,batchToSpaceND:ts,bincount:as,bitwiseAnd:En,booleanMaskAsync:xi,broadcastArgs:jn,broadcastTo:ss,buffer:nt,cast:rs,ceil:ns,clipByValue:is,clone:Nt,complex:os,concat:be,concat1d:us,concat2d:ps,concat3d:ls,concat4d:ms,conv1d:cs,conv2d:ds,conv2dTranspose:hs,conv3d:ys,conv3dTranspose:fs,cos:gs,cosh:bs,cosineWindow:Ns,cumprod:ws,cumsum:Ts,denseBincount:Ss,depthToSpace:vs,depthwiseConv2d:ht,diag:zn,dilation2d:ks,div:ct,divNoNan:Os,dot:_s,dropout:As,einsum:Is,elu:Es,enclosingPowerOfTwo:$s,ensureShape:Ln,equal:js,erf:Ds,euclideanNorm:zs,exp:xs,expandDims:Ls,expm1:Cs,eye:Vs,fft:Ps,fill:Fs,floor:Rs,floorDiv:Bs,fused:Ji,gather:mt,gatherND:Wi,greater:qs,greaterEqual:Ws,ifft:Hs,imag:Gs,image:wt,inTopKAsync:Gi,irfft:Us,isFinite:Ks,isInf:Js,isNaN:Xs,leakyRelu:Qs,less:Ys,lessEqual:Zs,linalg:Ms,linspace:Cn,localResponseNormalization:er,log:tr,log1p:ar,logSigmoid:sr,logSoftmax:rr,logSumExp:nr,logicalAnd:ir,logicalNot:or,logicalOr:ur,logicalXor:pr,losses:lr,lowerBound:Pn,matMul:H,max:mr,maxPool:cr,maxPool3d:dr,maxPoolWithArgmax:Rn,maximum:hr,mean:yr,meshgrid:Bn,min:fr,minimum:gr,mirrorPad:br,mod:Nr,moments:wr,movingAverage:Ci,mul:M,multiRNNCell:Wn,multinomial:Gn,neg:Tr,norm:Sr,notEqual:vr,oneHot:kr,ones:Z,onesLike:Or,op:T,outerProduct:Kn,pad:re,pad1d:Xn,pad2d:Yn,pad3d:Mn,pad4d:ti,pool:_r,pow:dt,prelu:Ar,print:Ir,prod:Er,raggedGather:si,raggedRange:ni,raggedTensorToTensor:oi,rand:pi,randomGamma:mi,randomNormal:it,randomStandardNormal:di,randomUniform:ot,randomUniformInt:yi,range:$r,real:jr,reciprocal:Dr,relu:zr,relu6:xr,reshape:k,reverse:ne,reverse1d:gi,reverse2d:Ni,reverse3d:Ti,reverse4d:vi,rfft:Lr,round:Cr,rsqrt:Vr,scalar:q,scatterND:Pi,searchSorted:Fe,selu:Pr,separableConv2d:Fr,setdiff1dAsync:Oi,sigmoid:le,sign:Rr,signal:Br,sin:qr,sinh:Wr,slice:G,slice1d:Hr,slice2d:Gr,slice3d:Ur,slice4d:Kr,softmax:Jr,softplus:Xr,spaceToBatchND:Qr,sparse:Yr,sparseToDense:Bi,spectral:Zr,split:Mr,sqrt:en,square:tn,squaredDifference:an,squeeze:lt,stack:se,step:sn,stridedSlice:rn,string:nn,sub:me,sum:on,tan:un,tanh:ke,tensor:Q,tensor1d:pn,tensor2d:ln,tensor3d:Sn,tensor4d:_i,tensor5d:Ai,tensor6d:Ii,tensorScatterUpdate:$i,tile:mn,topk:Tt,transpose:cn,truncatedNormal:dn,unique:hn,unsortedSegmentSum:yn,unstack:ie,upperBound:ji,variable:fn,where:gn,whereAsync:St,zeros:bn,zerosLike:Nn},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Jo=(a,e,t,s=I)=>{switch(a.op){case"BiasAdd":case"AddV2":case"Add":return[s.add(n("a",a,e,t),n("b",a,e,t))];case"AddN":return[s.addN(n("tensors",a,e,t))];case"FloorMod":case"Mod":return[s.mod(n("a",a,e,t),n("b",a,e,t))];case"Mul":return[s.mul(n("a",a,e,t),n("b",a,e,t))];case"RealDiv":case"Div":return[s.div(n("a",a,e,t),n("b",a,e,t))];case"DivNoNan":return[s.divNoNan(n("a",a,e,t),n("b",a,e,t))];case"FloorDiv":return[s.floorDiv(n("a",a,e,t),n("b",a,e,t))];case"Sub":return[s.sub(n("a",a,e,t),n("b",a,e,t))];case"Minimum":return[s.minimum(n("a",a,e,t),n("b",a,e,t))];case"Maximum":return[s.maximum(n("a",a,e,t),n("b",a,e,t))];case"Pow":return[s.pow(n("a",a,e,t),n("b",a,e,t))];case"SquaredDifference":return[s.squaredDifference(n("a",a,e,t),n("b",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const Xo=(a,e,t,s=I)=>{switch(a.op){case"Abs":case"ComplexAbs":return[s.abs(n("x",a,e,t))];case"Acos":return[s.acos(n("x",a,e,t))];case"Acosh":return[s.acosh(n("x",a,e,t))];case"Asin":return[s.asin(n("x",a,e,t))];case"Asinh":return[s.asinh(n("x",a,e,t))];case"Atan":return[s.atan(n("x",a,e,t))];case"Atan2":return[s.atan2(n("x",a,e,t),n("y",a,e,t))];case"Atanh":return[s.atanh(n("x",a,e,t))];case"Ceil":return[s.ceil(n("x",a,e,t))];case"Complex":return[s.complex(n("real",a,e,t),n("imag",a,e,t))];case"Cos":return[s.cos(n("x",a,e,t))];case"Cosh":return[s.cosh(n("x",a,e,t))];case"Elu":return[s.elu(n("x",a,e,t))];case"Erf":return[s.erf(n("x",a,e,t))];case"Exp":return[s.exp(n("x",a,e,t))];case"Expm1":return[s.expm1(n("x",a,e,t))];case"Floor":return[s.floor(n("x",a,e,t))];case"Log":return[s.log(n("x",a,e,t))];case"Log1p":return[s.log1p(n("x",a,e,t))];case"Imag":return[s.imag(n("x",a,e,t))];case"Neg":return[s.neg(n("x",a,e,t))];case"Reciprocal":return[s.reciprocal(n("x",a,e,t))];case"Real":return[s.real(n("x",a,e,t))];case"Relu":return[s.relu(n("x",a,e,t))];case"Round":return[s.round(n("x",a,e,t))];case"Selu":return[s.selu(n("x",a,e,t))];case"Sigmoid":return[s.sigmoid(n("x",a,e,t))];case"Sin":return[s.sin(n("x",a,e,t))];case"Sign":return[s.sign(n("x",a,e,t))];case"Sinh":return[s.sinh(n("x",a,e,t))];case"Softplus":return[s.softplus(n("x",a,e,t))];case"Sqrt":return[s.sqrt(n("x",a,e,t))];case"Square":return[s.square(n("x",a,e,t))];case"Tanh":return[s.tanh(n("x",a,e,t))];case"Tan":return[s.tan(n("x",a,e,t))];case"ClipByValue":return[s.clipByValue(n("x",a,e,t),n("clipValueMin",a,e,t),n("clipValueMax",a,e,t))];case"Relu6":return[s.relu6(n("x",a,e,t))];case"Rsqrt":return[s.rsqrt(A(a.inputNames[0],e,t))];case"LeakyRelu":return[s.leakyRelu(n("x",a,e,t),n("alpha",a,e,t))];case"Prelu":return[s.prelu(n("x",a,e,t),n("alpha",a,e,t))];case"IsNan":return[s.isNaN(A(a.inputNames[0],e,t))];case"IsInf":return[s.isInf(A(a.inputNames[0],e,t))];case"IsFinite":return[s.isFinite(A(a.inputNames[0],e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */function L(a,e,t=""){if(!(typeof a=="number"||typeof e=="number")){v(a.length===e.length,()=>t+` Shapes ${a} and ${e} must match`);for(let s=0;s<a.length;s++){const r=a[s],i=e[s];v(r<0||i<0||r===i,()=>t+` Shapes ${a} and ${e} must match`)}}}function Ze(a){return!(typeof a=="number"||a.some(e=>e<0))}function Y(a,e,t){let s=Le(a,t);const r=!Ze(s);if(r&&e.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${s}`);if(r&&e.forEach(i=>{s=Le(i.shape,s)}),!Ze(s))throw new Error(`Non-fully-defined elementShape: ${s}`);return s}function Le(a,e){if(typeof a=="number")return e;if(typeof e=="number")return a;if(a.length!==e.length)throw new Error(`Incompatible ranks during merge: ${a} vs. ${e}`);const t=[];for(let s=0;s<a.length;++s){const r=a[s],i=e[s];if(r>=0&&i>=0&&r!==i)throw new Error(`Incompatible shape during merge: ${a} vs. ${e}`);t[s]=r>=0?r:i}return t}/**
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
 */class Qo{constructor(e,t,s,r,i,u,o){this.name=e,this.dtype=t,this.maxSize=s,this.elementShape=r,this.identicalElementShapes=i,this.dynamicSize=u,this.clearAfterRead=o,this.tensors=[],this.closed_=!1,this.idTensor=q(0),B(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.tensor.id))&&t.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw new Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);const t=this.tensors[e];if(t.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(t.cleared=!0),t.read=!0,t.tensor}readMany(e){return e.map(t=>this.read(t))}write(e,t){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw new Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);const s=this.tensors[e]||{};if(t.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${t.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=t.shape),L(this.elementShape,t.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),s.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(s.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);s.tensor=t,B(t),s.written=!0,this.tensors[e]=s}writeMany(e,t){if(e.length!==t.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${t.length}.`);e.forEach((s,r)=>this.write(s,t[r]))}gather(e,t){if(t&&t!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${t}`);if(e)e=e.slice(0,this.size());else{e=[];for(let r=0;r<this.size();r++)e.push(r)}if(e.length===0)return Q([],[0].concat(this.elementShape));const s=this.readMany(e);return L(this.elementShape,s[0].shape,"TensorArray shape mismatch: "),se(s,0)}concat(e){if(e&&e!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(this.size()===0)return Q([],[0].concat(this.elementShape));const t=[];for(let r=0;r<this.size();r++)t.push(r);const s=this.readMany(t);return L(this.elementShape,s[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${s[0].shape})`),be(s,0)}scatter(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);if(e.length!==t.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);const s=Math.max(...e);if(!this.dynamicSize&&s>=this.maxSize)throw new Error(`Max index must be < array size (${s}  vs. ${this.maxSize})`);this.writeMany(e,ie(t,0))}split(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);let s=0;const r=e.map(p=>(s+=p,s));if(s!==t.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${t.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);const i=s===0?0:t.size/s,u=[];V(()=>{t=k(t,[1,s,i]);for(let p=0;p<e.length;++p){const m=[0,p===0?0:r[p-1],0],c=[1,e[p],i];u[p]=k(G(t,m,c),this.elementShape)}return u});const o=[];for(let p=0;p<e.length;p++)o[p]=p;this.writeMany(o,u)}}/**
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
 */class K{get id(){return this.idTensor.id}constructor(e,t,s,r=-1){this.tensors=e,this.elementShape=t,this.elementDtype=s,e?.forEach(i=>{if(s!==i.dtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${i.dtype}`);L(t,i.shape,"TensorList shape mismatch: "),B(i)}),this.idTensor=q(0),this.maxNumElements=r,B(this.idTensor)}copy(){return new K([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.id))&&t.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,t,s=-1){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(s!==-1&&this.tensors.length!==s)throw new Error(`Operation expected a list with ${s} elements but got a list with ${this.tensors.length} elements.`);L(e,this.elementShape,"TensorList shape mismatch: ");const r=Y(this.elementShape,this.tensors,e);return V(()=>{const i=this.tensors.map(u=>k(u,r));return se(i,0)})}popBack(e,t){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const s=Y(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,L(r.shape,e,"TensorList shape mismatch: "),k(r,s)}pushBack(e){if(e.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(L(e.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");B(e),this.tensors.push(e)}resize(e){if(e<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${e}`);if(this.maxNumElements!==-1&&e>this.maxNumElements)throw new Error(`TensorListResize input size ${e} is greater maxNumElement ${this.maxNumElements}.`);const t=new K([],this.elementShape,this.elementDtype,this.maxNumElements);t.tensors.length=e;for(let s=0;s<Math.min(this.tensors.length,e);++s)t.tensors[s]=this.tensors[s];return t}getItem(e,t,s){if(s!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw new Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(this.tensors[e]==null)throw new Error(`element at index ${e} is null.`);L(this.tensors[e].shape,t,"TensorList shape mismatch: ");const r=Y(this.elementShape,this.tensors,t);return k(this.tensors[e],r)}setItem(e,t){if(t.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(e<0||this.maxNumElements!==-1&&e>=this.maxNumElements)throw new Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);L(this.elementShape,t.shape,"TensorList shape mismatch: "),B(t),this.tensors[e]!=null&&(this.tensors[e].kept=!1),this.tensors[e]=t}gather(e,t,s){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);L(this.elementShape,s,"TensorList shape mismatch: "),e=e.slice(0,this.size());const r=Y(this.elementShape,this.tensors,s);return e.length===0?Q([],[0].concat(r)):V(()=>{const i=e.map(u=>k(this.tensors[u],r));return se(i,0)})}concat(e,t){if(e&&e!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);L(this.elementShape,t,"TensorList shape mismatch: ");const s=Y(this.elementShape,this.tensors,t);return this.size()===0?Q([],[0].concat(s)):V(()=>{const r=this.tensors.map(i=>k(i,s));return be(r,0)})}}function Yo(a,e,t){const s=a.dtype;if(a.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${a.shape}`);if(a.dtype!==t)throw new Error(`Invalid data types; op elements ${a.dtype}, but list elements ${t}`);const r=a.shape.slice(1);L(r,e,"TensorList shape mismatch: ");const i=ie(a);return new K(i,e,s)}function Zo(a,e,t,s){return new K([],a,e,s)}function Mo(a,e,t,s){if(e.length!==a.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${a.shape[0]}`);const r=Math.max(...e);if(s!=null&&s!==-1&&r>=s)throw new Error(`Max index must be < array size (${r}  vs. ${s})`);const i=new K([],t,a.dtype,s),u=ie(a,0);return e.forEach((o,p)=>{i.setItem(o,u[p])}),i}function eu(a,e,t){let s=0;const r=e.map(m=>(s+=m,s));if(s!==a.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${a.shape}`);const i=a.shape.slice(1),u=Le(i,t),o=s===0?0:a.size/s,p=V(()=>{const m=[];a=k(a,[1,s,o]);for(let c=0;c<e.length;++c){const y=[0,c===0?0:r[c-1],0],w=[1,e[c],o];m[c]=k(G(a,y,w),u)}return a.dispose(),m}),l=new K([],t,a.dtype,e.length);for(let m=0;m<p.length;m++)l.setItem(m,p[m]);return l}/**
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
 */const tu=async(a,e,t)=>{switch(a.op){case"If":case"StatelessIf":{const s=n("thenBranch",a,e,t),r=n("elseBranch",a,e,t),i=n("cond",a,e,t),u=n("args",a,e,t);return(await i.data())[0]?t.functionMap[s].executeFunctionAsync(u,t.tensorArrayMap,t.tensorListMap):t.functionMap[r].executeFunctionAsync(u,t.tensorArrayMap,t.tensorListMap)}case"While":case"StatelessWhile":{const s=n("body",a,e,t),r=n("cond",a,e,t),i=n("args",a,e,t),u=await t.functionMap[r].executeFunctionAsync(i,t.tensorArrayMap,t.tensorListMap),o=i.map(m=>m.id);let p=await u[0].data();u.forEach(m=>{!m.kept&&o.indexOf(m.id)===-1&&m.dispose()});let l=i;for(;p[0];){const m=l;l=await t.functionMap[s].executeFunctionAsync(l,t.tensorArrayMap,t.tensorListMap);const c=l.map(y=>y.id);m.forEach(y=>{!y.kept&&o.indexOf(y.id)===-1&&c.indexOf(y.id)===-1&&y.dispose()});const d=await t.functionMap[r].executeFunctionAsync(l,t.tensorArrayMap,t.tensorListMap);p=await d[0].data(),d.forEach(y=>{!y.kept&&o.indexOf(y.id)===-1&&c.indexOf(y.id)===-1&&y.dispose()})}return l}case"LoopCond":{const s=n("pred",a,e,t);return[R(s)]}case"Switch":{const s=n("pred",a,e,t);let r=n("data",a,e,t);return r.kept||(r=R(r)),(await s.data())[0]?[void 0,r]:[r,void 0]}case"Merge":{const s=a.inputNames.find(r=>A(r,e,t)!==void 0);if(s){const r=A(s,e,t);return[R(r)]}return}case"Enter":{const s=n("frameName",a,e,t),r=n("tensor",a,e,t);return t.enterFrame(s),[R(r)]}case"Exit":{const s=n("tensor",a,e,t);return t.exitFrame(),[R(s)]}case"NextIteration":{const s=n("tensor",a,e,t);return t.nextIteration(),[R(s)]}case"TensorArrayV3":{const s=n("size",a,e,t),r=n("dtype",a,e,t),i=n("elementShape",a,e,t),u=n("dynamicSize",a,e,t),o=n("clearAfterRead",a,e,t),p=n("identicalElementShapes",a,e,t),l=n("name",a,e,t),m=new Qo(l,r,s,i,p,u,o);return t.addTensorArray(m),[m.idTensor,q(1)]}case"TensorArrayWriteV3":{const s=n("tensorArrayId",a,e,t),r=n("index",a,e,t),i=n("tensor",a,e,t),u=t.getTensorArray(s.id);return u.write(r,i),[u.idTensor]}case"TensorArrayReadV3":{const s=n("tensorArrayId",a,e,t),r=n("index",a,e,t);return[t.getTensorArray(s.id).read(r)]}case"TensorArrayGatherV3":{const s=n("tensorArrayId",a,e,t),r=n("indices",a,e,t),i=n("dtype",a,e,t);return[t.getTensorArray(s.id).gather(r,i)]}case"TensorArrayScatterV3":{const s=n("tensorArrayId",a,e,t),r=n("indices",a,e,t),i=n("tensor",a,e,t),u=t.getTensorArray(s.id);return u.scatter(r,i),[u.idTensor]}case"TensorArrayConcatV3":{const s=n("tensorArrayId",a,e,t),r=t.getTensorArray(s.id),i=n("dtype",a,e,t);return[r.concat(i)]}case"TensorArraySplitV3":{const s=n("tensorArrayId",a,e,t),r=n("tensor",a,e,t),i=n("lengths",a,e,t),u=t.getTensorArray(s.id);return u.split(i,r),[u.idTensor]}case"TensorArraySizeV3":{const s=n("tensorArrayId",a,e,t),r=t.getTensorArray(s.id);return[q(r.size(),"int32")]}case"TensorArrayCloseV3":{const s=n("tensorArrayId",a,e,t),r=t.getTensorArray(s.id);return r.clearAndClose(),[r.idTensor]}case"TensorListSetItem":{const s=n("tensorListId",a,e,t),r=n("index",a,e,t),i=n("tensor",a,e,t),u=t.getTensorList(s.id);return u.setItem(r,i),[u.idTensor]}case"TensorListGetItem":{const s=n("tensorListId",a,e,t),r=n("index",a,e,t),i=n("elementShape",a,e,t),u=n("elementDType",a,e,t);return[t.getTensorList(s.id).getItem(r,i,u)]}case"TensorListScatterV2":case"TensorListScatter":{const s=n("indices",a,e,t),r=n("tensor",a,e,t),i=n("elementShape",a,e,t),u=n("numElements",a,e,t),o=Mo(r,s,i,u);return t.addTensorList(o),[o.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const s=n("elementShape",a,e,t),r=n("elementDType",a,e,t);let i;a.op==="TensorListReserve"?i="numElements":i="maxNumElements";const u=n(i,a,e,t),o=a.op==="TensorListReserve"?-1:u,p=Zo(s,r,u,o);return t.addTensorList(p),[p.idTensor]}case"TensorListGather":{const s=n("tensorListId",a,e,t),r=n("indices",a,e,t),i=n("elementShape",a,e,t),u=n("elementDType",a,e,t);return[t.getTensorList(s.id).gather(r,u,i)]}case"TensorListStack":{const s=n("tensorListId",a,e,t),r=n("elementShape",a,e,t),i=n("elementDType",a,e,t),u=n("numElements",a,e,t);return[t.getTensorList(s.id).stack(r,i,u)]}case"TensorListFromTensor":{const s=n("tensor",a,e,t),r=n("elementShape",a,e,t),i=n("elementDType",a,e,t),u=Yo(s,r,i);return t.addTensorList(u),[u.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const s=n("tensorListId",a,e,t),r=t.getTensorList(s.id),i=n("dtype",a,e,t),u=n("elementShape",a,e,t);return[r.concat(i,u)]}case"TensorListPushBack":{const s=n("tensorListId",a,e,t),r=n("tensor",a,e,t),i=t.getTensorList(s.id);return i.pushBack(r),[i.idTensor]}case"TensorListPopBack":{const s=n("tensorListId",a,e,t),r=n("elementShape",a,e,t),i=n("elementDType",a,e,t);return[t.getTensorList(s.id).popBack(r,i)]}case"TensorListSplit":{const s=n("tensor",a,e,t),r=n("elementShape",a,e,t),i=n("lengths",a,e,t),u=eu(s,i,r);return t.addTensorList(u),[u.idTensor]}case"TensorListLength":{const s=n("tensorListId",a,e,t),r=t.getTensorList(s.id);return[q(r.size(),"int32")]}case"TensorListResize":{const s=n("tensorListId",a,e,t),r=n("size",a,e,t),u=t.getTensorList(s.id).resize(r);return t.addTensorList(u),[u.idTensor]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */function Me(a,e,t){const[s,r]=n("fusedOps",a,e,t),i=s==="biasadd",u=!i,o=r==="prelu",p=s==="fusedbatchnorm",l=n("numArgs",a,e,t);if(i){if(o&&l!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!o&&i&&l!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(p)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const m=n("strides",a,e,t),c=ce(a,e,t),d=n("dataFormat",a,e,t).toUpperCase(),y=n("dilations",a,e,t);let[w,b]=n("args",a,e,t);u&&(b=w,w=void 0);const f=n("leakyreluAlpha",a,e,t);return{stride:m,pad:c,dataFormat:d,dilations:y,biasArg:w,preluArg:b,activationFunc:r,leakyreluAlpha:f}}const au=(a,e,t,s=I)=>{switch(a.op){case"Conv1D":{const r=n("stride",a,e,t),i=n("pad",a,e,t),u=n("dataFormat",a,e,t).toUpperCase(),o=n("dilation",a,e,t);return[s.conv1d(n("x",a,e,t),n("filter",a,e,t),r,i,u,o)]}case"Conv2D":{const r=n("strides",a,e,t),i=ce(a,e,t),u=n("dataFormat",a,e,t).toUpperCase(),o=n("dilations",a,e,t);return[s.conv2d(n("x",a,e,t),n("filter",a,e,t),[r[1],r[2]],i,u,[o[1],o[2]])]}case"_FusedConv2D":{const{stride:r,pad:i,dataFormat:u,dilations:o,biasArg:p,preluArg:l,activationFunc:m,leakyreluAlpha:c}=Me(a,e,t);return[s.fused.conv2d({x:n("x",a,e,t),filter:n("filter",a,e,t),strides:[r[1],r[2]],pad:i,dataFormat:u,dilations:[o[1],o[2]],bias:p,activation:m,preluActivationWeights:l,leakyreluAlpha:c})]}case"FusedDepthwiseConv2dNative":{const{stride:r,pad:i,dataFormat:u,dilations:o,biasArg:p,preluArg:l,activationFunc:m,leakyreluAlpha:c}=Me(a,e,t);return[s.fused.depthwiseConv2d({x:n("x",a,e,t),filter:n("filter",a,e,t),strides:[r[1],r[2]],pad:i,dataFormat:u,dilations:[o[1],o[2]],bias:p,activation:m,preluActivationWeights:l,leakyreluAlpha:c})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const r=n("outputShape",a,e,t),i=n("strides",a,e,t),u=ce(a,e,t);return[s.conv2dTranspose(n("x",a,e,t),n("filter",a,e,t),r,[i[1],i[2]],u)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const r=n("strides",a,e,t),i=ce(a,e,t),u=n("dilations",a,e,t),o=n("dataFormat",a,e,t).toUpperCase();return[s.depthwiseConv2d(n("input",a,e,t),n("filter",a,e,t),[r[1],r[2]],i,o,[u[1],u[2]])]}case"Conv3D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("dataFormat",a,e,t).toUpperCase(),o=n("dilations",a,e,t);return[s.conv3d(n("x",a,e,t),n("filter",a,e,t),[r[1],r[2],r[3]],i,u,[o[1],o[2],o[3]])]}case"AvgPool":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t);return[s.avgPool(n("x",a,e,t),[u[1],u[2]],[r[1],r[2]],i)]}case"MaxPool":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t);return[s.maxPool(n("x",a,e,t),[u[1],u[2]],[r[1],r[2]],i)]}case"MaxPoolWithArgmax":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t),o=n("includeBatchInIndex",a,e,t),{result:p,indexes:l}=s.maxPoolWithArgmax(n("x",a,e,t),[u[1],u[2]],[r[1],r[2]],i,o);return[p,l]}case"AvgPool3D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t);return[s.avgPool3d(n("x",a,e,t),[u[1],u[2],u[3]],[r[1],r[2],r[3]],i)]}case"MaxPool3D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t);return[s.maxPool3d(n("x",a,e,t),[u[1],u[2],u[3]],[r[1],r[2],r[3]],i)]}case"Dilation2D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("dilations",a,e,t),o=r[1],p=r[2],l=u[1],m=u[2];return[s.dilation2d(n("x",a,e,t),n("filter",a,e,t),[o,p],i,[l,m],"NHWC")]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const su=(a,e,t,s=I)=>{switch(a.op){case"Fill":{const r=n("shape",a,e,t),i=n("dtype",a,e,t),u=n("value",a,e,t);return[s.fill(r,u,i)]}case"LinSpace":{const r=n("start",a,e,t),i=n("stop",a,e,t),u=n("num",a,e,t);return[s.linspace(r,i,u)]}case"Multinomial":{const r=n("logits",a,e,t),i=n("numSamples",a,e,t),u=n("seed",a,e,t);return[s.multinomial(r,i,u)]}case"OneHot":{const r=n("indices",a,e,t),i=n("depth",a,e,t),u=n("onValue",a,e,t),o=n("offValue",a,e,t),p=n("dtype",a,e,t);return[s.oneHot(r,i,u,o,p)]}case"Ones":return[s.ones(n("shape",a,e,t),n("dtype",a,e,t))];case"OnesLike":return[s.onesLike(n("x",a,e,t))];case"RandomStandardNormal":return[s.randomStandardNormal(n("shape",a,e,t),n("dtype",a,e,t),n("seed",a,e,t))];case"RandomUniform":return[s.randomUniform(n("shape",a,e,t),n("minval",a,e,t),n("maxval",a,e,t),n("dtype",a,e,t))];case"RandomUniformInt":return[s.randomUniformInt(n("shape",a,e,t),n("minval",a,e,t),n("maxval",a,e,t),n("seed",a,e,t))];case"Range":{const r=n("start",a,e,t),i=n("stop",a,e,t),u=n("step",a,e,t);return[s.range(r,i,u,n("dtype",a,e,t))]}case"TruncatedNormal":{const r=n("shape",a,e,t),i=n("mean",a,e,t),u=n("stdDev",a,e,t),o=n("seed",a,e,t);return[s.truncatedNormal(r,i,u,n("dtype",a,e,t),o)]}case"Zeros":return[s.zeros(n("shape",a,e,t),n("dtype",a,e,t))];case"ZerosLike":return[s.zerosLike(n("x",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */function ve(a,e,t){const s=n("boxes",a,e,t),r=n("scores",a,e,t),i=n("maxOutputSize",a,e,t),u=n("iouThreshold",a,e,t),o=n("scoreThreshold",a,e,t),p=n("softNmsSigma",a,e,t);return{boxes:s,scores:r,maxOutputSize:i,iouThreshold:u,scoreThreshold:o,softNmsSigma:p}}const ru=async(a,e,t,s,r=I)=>{switch(a.op){case"NonMaxSuppressionV5":{const{boxes:i,scores:u,maxOutputSize:o,iouThreshold:p,scoreThreshold:l,softNmsSigma:m}=ve(a,e,t),c=await r.image.nonMaxSuppressionWithScoreAsync(i,u,o,p,l,m);return[c.selectedIndices,c.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:i,scores:u,maxOutputSize:o,iouThreshold:p,scoreThreshold:l}=ve(a,e,t),m=n("padToMaxOutputSize",a,e,t),c=await r.image.nonMaxSuppressionPaddedAsync(i,u,o,p,l,m);return[c.selectedIndices,c.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:i,scores:u,maxOutputSize:o,iouThreshold:p,scoreThreshold:l}=ve(a,e,t);return[await r.image.nonMaxSuppressionAsync(i,u,o,p,l)]}case"Where":{const i=r.cast(n("condition",a,e,t),"bool"),u=[await r.whereAsync(i)];return i.dispose(),u}case"ListDiff":return r.setdiff1dAsync(n("x",a,e,t),n("y",a,e,t));default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const nu=(a,e,t,s=I)=>{switch(a.op){case"LowerBound":{const r=n("sortedSequence",a,e,t),i=n("values",a,e,t);return[s.lowerBound(r,i)]}case"TopKV2":{const r=n("x",a,e,t),i=n("k",a,e,t),u=n("sorted",a,e,t),o=s.topk(r,i,u);return[o.values,o.indices]}case"UpperBound":{const r=n("sortedSequence",a,e,t),i=n("values",a,e,t);return[s.upperBound(r,i)]}case"Unique":{const r=n("x",a,e,t),i=s.unique(r);return[i.values,i.indices]}case"UniqueV2":{const r=n("x",a,e,t),i=n("axis",a,e,t),u=s.unique(r,i);return[u.values,u.indices]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const iu=(a,e,t,s=I)=>{switch(a.op){case"Const":return e[a.name];case"PlaceholderWithDefault":const r=n("default",a,e,t);return[A(a.name,e,t)||r];case"Placeholder":return[A(a.name,e,t)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const m=n("x",a,e,t);return[R(m)]}case"IdentityN":return n("x",a,e,t).map(m=>R(m));case"Snapshot":const i=n("x",a,e,t);return[R(i)];case"Shape":return[s.tensor1d(n("x",a,e,t).shape,"int32")];case"ShapeN":return n("x",a,e,t).map(m=>s.tensor1d(m.shape));case"Size":return[s.scalar(n("x",a,e,t).size,"int32")];case"Rank":return[s.scalar(n("x",a,e,t).rank,"int32")];case"NoOp":return[s.scalar(1)];case"Print":const u=n("x",a,e,t),o=n("data",a,e,t),p=n("message",a,e,t),l=n("summarize",a,e,t);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(p);for(let m=0;m<o.length;m++)console.log(Array.prototype.slice.call(o[m].dataSync()).slice(0,l));return[u];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */class ou{get id(){return this.handle.id}constructor(e,t){this.keyDType=e,this.valueDType=t,this.handle=q(0),this.tensorMap=new Map,B(this.handle)}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return q(this.size(),"int32")}async import(e,t){this.checkKeyAndValueTensor(e,t);const s=await e.data();return this.tensorMap.forEach(r=>r.dispose()),this.tensorMap.clear(),V(()=>{const r=ie(t),i=s.length,u=r.length;v(i===u,()=>`The number of elements doesn't match, keys has ${i} elements, the values has ${u} elements.`);for(let o=0;o<i;o++){const p=s[o],l=r[o];B(l),this.tensorMap.set(p,l)}return this.handle})}async find(e,t){this.checkKeyAndValueTensor(e,t);const s=await e.data();return V(()=>{const r=[];for(let i=0;i<s.length;i++){const u=s[i],o=this.findWithDefault(u,t);r.push(o)}return se(r)})}findWithDefault(e,t){const s=this.tensorMap.get(e);return s??t}checkKeyAndValueTensor(e,t){if(e.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(t.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${t.dtype}`)}}/**
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
 */const uu=async(a,e,t,s)=>{switch(a.op){case"HashTable":case"HashTableV2":{const r=s.getHashTableHandleByName(a.name);if(r!=null)return[r];{const i=n("keyDType",a,e,t),u=n("valueDType",a,e,t),o=new ou(i,u);return s.addHashTable(a.name,o),[o.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const r=n("tableHandle",a,e,t,s),i=n("keys",a,e,t),u=n("values",a,e,t);return[await s.getHashTableById(r.id).import(i,u)]}case"LookupTableFind":case"LookupTableFindV2":{const r=n("tableHandle",a,e,t,s),i=n("keys",a,e,t),u=n("defaultValue",a,e,t);return[await s.getHashTableById(r.id).find(i,u)]}case"LookupTableSize":case"LookupTableSizeV2":{const r=n("tableHandle",a,e,t,s);return[s.getHashTableById(r.id).tensorSize()]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const pu=(a,e,t,s=I)=>{switch(a.op){case"ResizeBilinear":{const r=n("images",a,e,t),i=n("size",a,e,t),u=n("alignCorners",a,e,t),o=n("halfPixelCenters",a,e,t);return[s.image.resizeBilinear(r,[i[0],i[1]],u,o)]}case"ResizeNearestNeighbor":{const r=n("images",a,e,t),i=n("size",a,e,t),u=n("alignCorners",a,e,t),o=n("halfPixelCenters",a,e,t);return[s.image.resizeNearestNeighbor(r,[i[0],i[1]],u,o)]}case"CropAndResize":{const r=n("image",a,e,t),i=n("boxes",a,e,t),u=n("boxInd",a,e,t),o=n("cropSize",a,e,t),p=n("method",a,e,t),l=n("extrapolationValue",a,e,t);return[s.image.cropAndResize(r,i,u,o,p,l)]}case"ImageProjectiveTransformV3":{const r=n("images",a,e,t),i=n("transforms",a,e,t),u=n("outputShape",a,e,t),o=n("fillValue",a,e,t),p=n("interpolation",a,e,t),l=n("fillMode",a,e,t);return[s.image.transform(r,i,p.toLowerCase(),l.toLowerCase(),o,u)]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const lu=(a,e,t,s=I)=>{switch(a.op){case"Equal":return[s.equal(n("a",a,e,t),n("b",a,e,t))];case"NotEqual":return[s.notEqual(n("a",a,e,t),n("b",a,e,t))];case"Greater":return[s.greater(n("a",a,e,t),n("b",a,e,t))];case"GreaterEqual":return[s.greaterEqual(n("a",a,e,t),n("b",a,e,t))];case"Less":return[s.less(n("a",a,e,t),n("b",a,e,t))];case"LessEqual":return[s.lessEqual(n("a",a,e,t),n("b",a,e,t))];case"LogicalAnd":return[s.logicalAnd(n("a",a,e,t),n("b",a,e,t))];case"LogicalNot":return[s.logicalNot(n("a",a,e,t))];case"LogicalOr":return[s.logicalOr(n("a",a,e,t),n("b",a,e,t))];case"Select":case"SelectV2":return[s.where(n("condition",a,e,t),n("a",a,e,t),n("b",a,e,t))];case"BitwiseAnd":return[s.bitwiseAnd(n("a",a,e,t),n("b",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const mu=(a,e,t,s=I)=>{switch(a.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[s.matMul(n("a",a,e,t),n("b",a,e,t),n("transposeA",a,e,t),n("transposeB",a,e,t))];case"Einsum":return[s.einsum(n("equation",a,e,t),...n("tensors",a,e,t))];case"Transpose":return[s.transpose(n("x",a,e,t),n("perm",a,e,t))];case"_FusedMatMul":const[r,i]=n("fusedOps",a,e,t),u=r==="biasadd",o=i==="prelu",p=n("numArgs",a,e,t),l=n("leakyreluAlpha",a,e,t);if(u){if(o&&p!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!o&&p!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[m,c]=n("args",a,e,t);return[s.fused.matMul({a:n("a",a,e,t),b:n("b",a,e,t),transposeA:n("transposeA",a,e,t),transposeB:n("transposeB",a,e,t),bias:m,activation:i,preluActivationWeights:c,leakyreluAlpha:l})];case"MatrixBandPart":return[s.linalg.bandPart(n("a",a,e,t),n("numLower",a,e,t),n("numUpper",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const cu=(a,e,t,s=I)=>{switch(a.op){case"EuclideanNorm":return[s.euclideanNorm(n("x",a,e,t),n("axis",a,e,t),n("keepDims",a,e,t))];case"FusedBatchNorm":case"FusedBatchNormV2":return[s.batchNorm(n("x",a,e,t),n("mean",a,e,t),n("variance",a,e,t),n("offset",a,e,t),n("scale",a,e,t),n("epsilon",a,e,t))];case"FusedBatchNormV3":return[s.batchNorm(n("x",a,e,t),n("mean",a,e,t),n("variance",a,e,t),n("offset",a,e,t),n("scale",a,e,t),n("epsilon",a,e,t))];case"LRN":return[s.localResponseNormalization(n("x",a,e,t),n("radius",a,e,t),n("bias",a,e,t),n("alpha",a,e,t),n("beta",a,e,t))];case"Softmax":return[s.softmax(n("x",a,e,t))];case"LogSoftmax":return[s.logSoftmax(n("x",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const du=(a,e,t,s=I)=>{switch(a.op){case"RaggedGather":{const{outputNestedSplits:r,outputDenseValues:i}=s.raggedGather(n("paramsNestedSplits",a,e,t),n("paramsDenseValues",a,e,t),n("indices",a,e,t),n("outputRaggedRank",a,e,t));return r.concat(i)}case"RaggedRange":{const{rtNestedSplits:r,rtDenseValues:i}=s.raggedRange(n("starts",a,e,t),n("limits",a,e,t),n("splits",a,e,t));return[r,i]}case"RaggedTensorToTensor":return[s.raggedTensorToTensor(n("shape",a,e,t),n("values",a,e,t),n("defaultValue",a,e,t),n("rowPartitionTensors",a,e,t),n("rowPartitionTypes",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const hu=(a,e,t,s=I)=>{switch(a.op){case"Max":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.max(n("x",a,e,t),o,p)]}case"Mean":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.mean(n("x",a,e,t),o,p)]}case"Min":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.min(n("x",a,e,t),o,p)]}case"Sum":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.sum(n("x",a,e,t),o,p)]}case"All":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.all(n("x",a,e,t),o,p)]}case"Any":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.any(n("x",a,e,t),o,p)]}case"ArgMax":{const o=n("axis",a,e,t);return[s.argMax(n("x",a,e,t),o)]}case"ArgMin":{const o=n("axis",a,e,t);return[s.argMin(n("x",a,e,t),o)]}case"Prod":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.prod(n("x",a,e,t),o,p)]}case"Cumprod":{const o=n("axis",a,e,t),p=n("exclusive",a,e,t),l=n("reverse",a,e,t);return[s.cumprod(n("x",a,e,t),o,p,l)]}case"Cumsum":{const o=n("axis",a,e,t),p=n("exclusive",a,e,t),l=n("reverse",a,e,t);return[s.cumsum(n("x",a,e,t),o,p,l)]}case"Bincount":const r=n("x",a,e,t),i=n("weights",a,e,t),u=n("size",a,e,t);return[s.bincount(r,i,u)];case"DenseBincount":{const o=n("x",a,e,t),p=n("weights",a,e,t),l=n("size",a,e,t),m=n("binaryOutput",a,e,t);return[s.denseBincount(o,p,l,m)]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const yu=(a,e,t,s=I)=>{switch(a.op){case"ConcatV2":case"Concat":{const r=n("n",a,e,t),i=n("axis",a,e,t);let u=n("tensors",a,e,t);return u=u.slice(0,r),[s.concat(u,i)]}case"Gather":{const r=n("x",a,e,t),i=n("indices",a,e,t);return[s.gather(r,s.cast(i,"int32"),0)]}case"GatherV2":{const r=n("axis",a,e,t),i=n("batchDims",a,e,t),u=n("x",a,e,t),o=n("indices",a,e,t);return[s.gather(u,s.cast(o,"int32"),r,i)]}case"Reverse":{const r=n("dims",a,e,t),i=[];for(let o=0;o<r.length;o++)r[o]&&i.push(o);const u=n("x",a,e,t);return[s.reverse(u,i)]}case"ReverseV2":{const r=n("axis",a,e,t),i=n("x",a,e,t);return[s.reverse(i,r)]}case"Slice":{const r=n("begin",a,e,t),i=n("size",a,e,t);return[s.slice(n("x",a,e,t),r,i)]}case"StridedSlice":{const r=n("begin",a,e,t),i=n("end",a,e,t),u=n("strides",a,e,t),o=n("beginMask",a,e,t),p=n("endMask",a,e,t),l=n("ellipsisMask",a,e,t),m=n("newAxisMask",a,e,t),c=n("shrinkAxisMask",a,e,t),d=n("x",a,e,t);return[s.stridedSlice(d,r,i,u,o,p,l,m,c)]}case"Pack":return V(()=>{const r=n("axis",a,e,t),i=n("tensors",a,e,t),u=i[0].shape,o=s.squeeze(i[0]).shape,p=i.map(l=>{const m=ee(l.shape,u);if(!m&&!ee(s.squeeze(l).shape,o))throw new Error("the input tensors shape does not match");return m?l:s.reshape(l,u)});return[s.stack(p,r)]});case"Unpack":{const r=n("axis",a,e,t),i=n("tensor",a,e,t);return s.unstack(i,r)}case"Tile":{const r=n("reps",a,e,t);return[s.tile(n("x",a,e,t),r)]}case"Split":case"SplitV":{const r=n("axis",a,e,t),i=n("numOrSizeSplits",a,e,t),u=n("x",a,e,t);return s.split(u,i,r)}case"ScatterNd":{const r=n("indices",a,e,t),i=n("values",a,e,t),u=n("shape",a,e,t);return[s.scatterND(r,i,u)]}case"GatherNd":{const r=n("x",a,e,t),i=n("indices",a,e,t);return[s.gatherND(r,i)]}case"SparseToDense":{const r=n("sparseIndices",a,e,t),i=n("outputShape",a,e,t),u=n("sparseValues",a,e,t),o=n("defaultValue",a,e,t);return[s.sparseToDense(r,u,i,u.dtype===o.dtype?o:s.cast(o,u.dtype))]}case"TensorScatterUpdate":{const r=n("indices",a,e,t),i=n("values",a,e,t),u=n("tensor",a,e,t);return[s.tensorScatterUpdate(u,r,i)]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const fu=(a,e,t,s=I)=>{switch(a.op){case"SparseFillEmptyRows":{const{outputIndices:r,outputValues:i,emptyRowIndicator:u,reverseIndexMap:o}=s.sparse.sparseFillEmptyRows(n("indices",a,e,t),n("values",a,e,t),n("denseShape",a,e,t),n("defaultValue",a,e,t));return[r,i,u,o]}case"SparseReshape":{const{outputIndices:r,outputShape:i}=s.sparse.sparseReshape(n("inputIndices",a,e,t),n("inputShape",a,e,t),n("newShape",a,e,t));return[r,i]}case"SparseSegmentMean":return[s.sparse.sparseSegmentMean(n("data",a,e,t),n("indices",a,e,t),n("segmentIds",a,e,t))];case"SparseSegmentSum":return[s.sparse.sparseSegmentSum(n("data",a,e,t),n("indices",a,e,t),n("segmentIds",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const gu=(a,e,t,s=I)=>{switch(a.op){case"FFT":return[s.fft(n("x",a,e,t))];case"IFFT":return[s.ifft(n("x",a,e,t))];case"RFFT":return[s.rfft(n("x",a,e,t))];case"IRFFT":return[s.irfft(n("x",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const bu=(a,e,t,s=I)=>{switch(a.op){case"StaticRegexReplace":return[s.string.staticRegexReplace(n("input",a,e,t),n("pattern",a,e,t),n("rewrite",a,e,t),n("replaceGlobal",a,e,t))];case"StringNGrams":{const{nGrams:r,nGramsSplits:i}=s.string.stringNGrams(n("data",a,e,t),n("dataSplits",a,e,t),n("separator",a,e,t),n("nGramWidths",a,e,t),n("leftPad",a,e,t),n("rightPad",a,e,t),n("padWidth",a,e,t),n("preserveShortSequences",a,e,t));return[r,i]}case"StringSplit":{const{indices:r,values:i,shape:u}=s.string.stringSplit(n("input",a,e,t),n("delimiter",a,e,t),n("skipEmpty",a,e,t));return[r,i,u]}case"StringToHashBucketFast":return[s.string.stringToHashBucketFast(n("input",a,e,t),n("numBuckets",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const Nu=(a,e,t,s=I)=>{switch(a.op){case"Cast":return[s.cast(n("x",a,e,t),n("dtype",a,e,t))];case"ExpandDims":{const r=n("axis",a,e,t);return[s.expandDims(n("x",a,e,t),r)]}case"Squeeze":{const r=n("axis",a,e,t);return[s.squeeze(n("x",a,e,t),r)]}case"Reshape":return[s.reshape(n("x",a,e,t),n("shape",a,e,t))];case"EnsureShape":return[s.ensureShape(n("x",a,e,t),n("shape",a,e,t))];case"MirrorPad":return[s.mirrorPad(n("x",a,e,t),n("padding",a,e,t),n("mode",a,e,t))];case"PadV2":case"Pad":return[s.pad(n("x",a,e,t),n("padding",a,e,t),n("constantValue",a,e,t))];case"SpaceToBatchND":{const r=n("blockShape",a,e,t),i=n("paddings",a,e,t);return[s.spaceToBatchND(n("x",a,e,t),r,i)]}case"BatchToSpaceND":{const r=n("blockShape",a,e,t),i=n("crops",a,e,t);return[s.batchToSpaceND(n("x",a,e,t),r,i)]}case"DepthToSpace":{const r=n("blockSize",a,e,t),i=n("dataFormat",a,e,t).toUpperCase();return[s.depthToSpace(n("x",a,e,t),r,i)]}case"BroadcastTo":return[s.broadcastTo(n("x",a,e,t),n("shape",a,e,t))];case"BroadcastArgs":return[s.broadcastArgs(n("s0",a,e,t),n("s1",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */function et(a,e,t,s,r=V){const i=((u,o,p)=>{switch(u.category){case"arithmetic":return r(()=>Jo(u,o,p));case"basic_math":return r(()=>Xo(u,o,p));case"control":return tu(u,o,p);case"convolution":return r(()=>au(u,o,p));case"creation":return r(()=>su(u,o,p));case"dynamic":return ru(u,o,p);case"evaluation":return r(()=>nu(u,o,p));case"image":return r(()=>pu(u,o,p));case"graph":return r(()=>iu(u,o,p));case"logical":return r(()=>lu(u,o,p));case"matrices":return r(()=>mu(u,o,p));case"normalization":return r(()=>cu(u,o,p));case"ragged":return r(()=>du(u,o,p));case"reduction":return r(()=>hu(u,o,p));case"slice_join":return r(()=>yu(u,o,p));case"sparse":return r(()=>fu(u,o,p));case"spectral":return r(()=>gu(u,o,p));case"string":return r(()=>bu(u,o,p));case"transformation":return r(()=>Nu(u,o,p));case"hash_table":return uu(u,o,p,s);case"custom":const l=_t(u.op);if(l&&l.customExecutor)return l.customExecutor(new Ko(u,o,p));throw TypeError(`Custom op ${u.op} is not registered.`);default:throw TypeError(`Unknown op '${u.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(a,e,t);return ye(i)?i.then(u=>[].concat(u)):[].concat(i)}class tt{constructor(e={},t={},s={},r={},i){this.weightMap=e,this.tensorArrayMap=t,this.tensorListMap=s,this.functionMap=r,this.parseNodeNameCache=i,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,t){return{id:e,frameName:t,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const e=[];for(let t=0;t<this.contexts.length-1;t++){const s=this.contexts.slice(0,this.contexts.length-t);e.push(this.contextIdforContexts(s))}e.push(""),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(t=>t.id===0&&t.iterationId===0?"":`${t.frameName}-${t.iterationId}`).join("/"):""}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(const t in this.tensorArrayMap)this.tensorArrayMap[t].clearAndClose(e);for(const t in this.tensorListMap)this.tensorListMap[t].clearAndClose(e)}}/**
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
 */function at(a,e,t,s){const r=new Set,i=[];let u=null,o=null;const p=new Set,l=new Set(Object.keys(a).map(d=>j(d)[0]));s=s||[];const m=new Set(s.map(d=>j(d.name)[0])),c=[...e];for(;c.length>0;){const d=c.pop();if((W(d)||Au(d)||Iu(d))&&u==null&&(u=d,o=u.children.map(y=>y.name).filter(y=>r.has(y))),r.add(d.name),t[d.name]==null&&!l.has(d.name)&&!m.has(d.name)){if(d.inputs.length===0){i.push(d.name);continue}d.inputs.forEach(y=>{p.has(y.name)||(p.add(y.name),c.push(y))})}}return{inputs:a,outputs:e,usedNodes:r,missingInputs:i,dynamicNode:u,syncInputs:o}}function wu(a,e){const{usedNodes:t,inputs:s}=e,r=Object.keys(s).map(f=>j(f)[0]).map(f=>a.nodes[f]),i=a.initNodes||[],u=f=>t.has(typeof f=="string"?f:f.name);function o(f){return[...new Map(f.map(N=>[N.name,N])).values()]}const p=o([...r,...a.weights,...i]).filter(u),l=o([...p,...Object.values(a.nodes)]).filter(u),m=new Map(l.map(f=>[f.name,f])),c={};for(const f of l){c[f.name]=c[f.name]||0;for(const N of f.children)u(N)||(c[N.name]=Number.POSITIVE_INFINITY),c[N.name]=(c[N.name]||0)+1}const d=Object.entries(c).filter(([,f])=>f===0).map(([f])=>f),y=[...d];for(;d.length>0;){const f=d.pop(),N=m.get(f);for(const O of N.children.filter(u))--c[O.name]===0&&(y.push(O.name),d.push(O.name))}const w=y.map(f=>m.get(f)),b=Tu(w,p);return Su(b,p),b}function Tu(a,e){const t=new Map(a.map(u=>[u.name,u])),s=e.map(u=>u.name),r=new Set(s);for(;s.length>0;){const u=s.pop(),o=t.get(u);for(const p of o.children)!t.has(p.name)||r.has(p.name)||(r.add(p.name),s.push(p.name))}return a.filter(u=>r.has(u.name))}class ue extends Error{constructor(e){super(`NodesExecutionOrderError: ${e}`)}}function Su(a,e){const t=new Map(a.map((o,p)=>[o.name,p])),s=new Set(e.map(o=>o.name)),r=o=>s.has(typeof o=="string"?o:o.name),i=new Set(a.map(o=>o.name)),u=o=>i.has(typeof o=="string"?o:o.name);for(const o of a){for(const p of o.children.filter(u)){if(!t.has(p.name))throw new ue(`Child ${p.name} of node ${o.name} is unreachable.`);if(t.get(o.name)>t.get(p.name))throw new ue(`Node ${o.name} is scheduled to run after its child ${p.name}.`)}if(!r(o))for(const p of o.inputs){if(!t.has(p.name))throw new ue(`Input ${p.name} of node ${o.name} is unreachable.`);if(t.get(p.name)>t.get(o.name))throw new ue(`Node ${o.name} is scheduled to run before its input ${p.name}.`)}}}function vu(a){const e=new Map(a.map((o,p)=>[o.name,p])),t=Number.MAX_SAFE_INTEGER,s=a.map((o,p)=>W(o)?t:p),r=o=>{const p=s[e.get(o.name)];return p??-1},i=a.map((o,p)=>o.children.map(r).reduce((l,m)=>Math.max(l,m),s[p])),u=new Map;for(let o=0;o<a.length;++o){const p=i[o];if(p===t)continue;const l=a[o],m=a[p];u.has(m.name)||u.set(m.name,[]),u.get(m.name).push(l)}return u}const ku=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),Ou=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),_u=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function W(a){return ku.has(a.op)}function Au(a){return Ou.has(a.op)}function Iu(a){return _u.has(a.op)}/**
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
 */class ge{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){const t=Object.keys(e).map(s=>e[s].map(r=>r.id));this._weightIds=[].concat(...t),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{const t=e.signatureKey||e.name;return e.defaultOutput?`${t}:${e.defaultOutput}`:t})}get functions(){return Object.keys(this._functions).reduce((e,t)=>(e[t]=this._functions[t].signature,e),{})}constructor(e,t){this.graph=e,this.parent=t,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=e.outputs,this._inputs=e.inputs,this._initNodes=e.initNodes,this._signature=e.signature,this._functions=e.functions,e.functions!=null&&Object.keys(e.functions).forEach(s=>{this._functionExecutorMap[s]=new ge(e.functions[s],this)})}getCompilationKey(e,t){const s=e.map(i=>i.name).sort(),r=t.map(i=>i.name).sort();return s.join(this.SEPARATOR)+"--"+r.join(this.SEPARATOR)}compile(e,t){const s=at(e,t,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:i,syncInputs:u}=s;if(i!=null)throw new Error(`This execution contains the node '${i.name}', which has the dynamic op '${i.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${u}]`);if(r.length>0){const l=t.map(c=>c.name),m=Object.keys(e);throw new Error(`Cannot compute the outputs [${l}] from the provided inputs [${m}]. Missing the following inputs: [${r}]`)}const o=wu(this.graph,s),p=vu(o);return{orderedNodes:o,nodeLiveUntilMap:p}}cloneAndKeepTensor(e){if(e==null)return null;const t=e.clone();return B(t),t}cloneTensorList(e){return e?e.map(s=>this.cloneAndKeepTensor(s)):null}cloneTensorMap(e){return Object.fromEntries(Object.entries(e).map(([t,s])=>[t,this.cloneTensorList(s)]))}execute(e,t){this.disposeIntermediateTensors(),e=this.mapInputs(e);const s=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t);const r=s.map(d=>this.graph.nodes[j(d)[0]]),i=t.map(d=>j(d)[0]),u=new Set(i);let o=i.map(d=>this.graph.nodes[d]);o.length===0&&(o=this._outputs);const p=this.getCompilationKey(r,o);let l=this.compiledMap.get(p);l==null&&(l=this.compile(e,o),this.compiledMap.set(p,l));try{this.keepIntermediateTensors=ae().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(d){this.keepIntermediateTensors=!1,console.warn(d.message)}const m={},c={};return V(()=>{const d=new tt(this.weightMap,m,c,this.functionExecutorMap,this.parseNodeNameCache),y=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(e).forEach(N=>{const[O,$]=j(N,d),S=[];S[$]=e[N],y[O]=S,this.keepIntermediateTensors&&(this.clonedTensorsMap[O]=this.cloneTensorList(S))});const w=this.getFrozenTensorIds(y),{orderedNodes:b,nodeLiveUntilMap:f}=l;for(const N of b){if(y[N.name])continue;const O=et(N,y,d,this._resourceManager);if(ye(O))throw new Error(`The execution of the op '${N.op}' returned a promise. Please use model.executeAsync() instead.`);y[N.name]=O,this.keepIntermediateTensors&&(this.clonedTensorsMap[N.name]=this.cloneTensorList(O)),this.checkTensorForDisposalWithNodeLiveUntilInfo(N,y,d,w,u,f.get(N.name))}return this.parent==null&&d.dispose(w),t.map(N=>A(N,y,d))})}getFrozenTensorIds(e){const t=[].concat.apply([],Object.keys(e).map(s=>e[s]).map(s=>s.map(r=>r.id)));return new Set(t)}checkTensorForDisposal(e,t,s,r,i,u,o){if(!(W(t)||u.has(e))){for(const p of s[e])p!=null&&(o[p.id]=(o[p.id]||0)+t.children.length);for(const p of t.inputs){if(W(p))continue;const l=Xe(p.name,s,r);if(l!=null)for(const m of l){if(!m||m.kept||i.has(m.id))continue;const c=o[m.id];c===1?(m.dispose(),delete o[m.id]):c!=null&&o[m.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(e,t,s,r,i,u){function o(p){return W(p)||i.has(p.name)}if(!(W(e)||u==null))for(const p of u){if(o(p))continue;const l=Xe(p.name,t,s);for(const m of l)!m||m.kept||r.has(m.id)||m.dispose()}}async executeAsync(e,t){return this._executeAsync(e,t)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(e=>{for(const t of e)t&&!t.isDisposed&&t.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(e,t,s=!1,r={},i={}){this.disposeIntermediateTensors(),s||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t));try{this.keepIntermediateTensors=ae().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(d){this.keepIntermediateTensors=!1,console.warn(d.message)}const u=new tt(this.weightMap,r,i,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const o=await this.executeWithControlFlow(e,u,t,s),p=t.map(d=>A(d,o,u)),l=p.map(d=>d.id),m=Object.keys(e).map(d=>e[d].id),c=new Set([...l,...m,...this.weightIds]);return Object.values(o).forEach(d=>{d.forEach(y=>{y&&!y.isDisposed&&!c.has(y.id)&&y.dispose()})}),this.parent==null&&u.dispose(c),p}async executeFunctionAsync(e,t,s){const r=e.reduce((i,u,o)=>(i[this.inputs[o].name]=u,i),{});return this._executeAsync(r,this.outputNodes,!0,t,s)}async executeWithControlFlow(e,t,s,r){const i=Object.keys(e),u=i.map(S=>this.graph.nodes[j(S)[0]]),o=s.map(S=>j(S)[0]),p=new Set(o);let l=o.map(S=>this.graph.nodes[S]);l.length===0&&(l=this._outputs);const{usedNodes:m,missingInputs:c,dynamicNode:d,syncInputs:y}=at(e,l,this.weightMap,this._initNodes),w=[...u,...this.graph.weights,...this._initNodes||[]].map(S=>({node:S,contexts:t.currentContext})),b=Object.assign({},this.weightMap);Object.keys(e).forEach(S=>{const[E,D]=j(S),z=[];z[D]=e[S],b[E]=z});const f={},N=this.getFrozenTensorIds(b),O={};for(;w.length>0;){const S=this.processStack(u,w,t,b,O,N,p,f,m);await Promise.all(S)}d==null&&!r&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const $=l.filter(S=>!W(S)&&!A(S.name,b,t)).map(S=>S.name);if($.length>0){let S="";throw d!=null&&(S=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${y}]`),new Error(`Cannot compute the outputs [${$}] from the provided inputs [${i}]. Consider providing the following inputs: [${c}]. ${S}`)}return b}processStack(e,t,s,r,i,u,o,p,l){const m=[];for(;t.length>0;){const c=t.pop();s.currentContext=c.contexts;let d="";if(c.node.op==="Enter"&&n("isConstant",c.node,r,s)&&([d]=F(c.node.name,s)),r[c.node.name]==null){const y=et(c.node,r,s,this._resourceManager);d||([d]=F(c.node.name,s));const w=s.currentContext;ye(y)?m.push(y.then(b=>(r[d]=b,this.keepIntermediateTensors&&(this.clonedTensorsMap[d]=this.cloneTensorList(b)),s.currentContext=w,this.checkTensorForDisposal(d,c.node,r,s,u,o,p),this.processChildNodes(c.node,t,s,r,i,l),b))):(r[d]=y,this.keepIntermediateTensors&&(this.clonedTensorsMap[d]=this.cloneTensorList(y)),this.checkTensorForDisposal(d,c.node,r,s,u,o,p),this.processChildNodes(c.node,t,s,r,i,l))}else this.processChildNodes(c.node,t,s,r,i,l)}return m}processChildNodes(e,t,s,r,i,u){e.children.forEach(o=>{const[p]=F(o.name,s);i[p]||!u.has(o.name)||(o.op==="Merge"?o.inputNames.some(l=>!!A(l,r,s))&&(i[p]=!0,t.push({contexts:s.currentContext,node:o})):o.inputNames.every(l=>!!A(l,r,s))&&(i[p]=!0,t.push({contexts:s.currentContext,node:o})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(t=>t.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(t=>{const s=e[t],[r]=j(t),i=this.graph.nodes[r];if(i.attrParams.shape&&i.attrParams.shape.value){const u=i.attrParams.shape.value,o=u.length===s.shape.length&&s.shape.every((p,l)=>u[l]===-1||u[l]===p);v(o,()=>`The shape of dict['${i.name}'] provided in model.execute(dict) must be [${u}], but was [${s.shape}]`)}i.attrParams.dtype&&i.attrParams.dtype.value&&v(s.dtype===i.attrParams.dtype.value,()=>`The dtype of dict['${i.name}'] provided in model.execute(dict) must be ${i.attrParams.dtype.value}, but was ${s.dtype}`)})}mapInputs(e){var t,s;const r={};for(const i in e){const u=(s=(t=this._signature)===null||t===void 0?void 0:t.inputs)===null||s===void 0?void 0:s[i];u!=null?r[u.name]=e[i]:r[i]=e[i]}return r}checkInputs(e){const t=Object.keys(e).filter(s=>{const[r]=j(s);return this.graph.nodes[r]==null});if(t.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${t}] that are not part of graph`)}mapOutputs(e){return e.map(t=>{var s,r;const i=(r=(s=this._signature)===null||s===void 0?void 0:s.outputs)===null||r===void 0?void 0:r[t];return i!=null?i.name:t},{})}checkOutputs(e){e.forEach(t=>{const[s]=j(t);if(!this.graph.nodes[s])throw new Error(`The output '${t}' is not found in the graph`)})}}class Eu{constructor(e={},t={}){this.hashTableNameToHandle=e,this.hashTableMap=t}addHashTable(e,t){this.hashTableNameToHandle[e]=t.handle,this.hashTableMap[t.id]=t}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(const e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(const e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}}/**
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
 */const $u="?tfjs-format=file",ju="model.json";class Du{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(e,t={},s=Ot){this.modelUrl=e,this.loadOptions=t,this.version="n/a",this.io=s,t==null&&(this.loadOptions={}),this.resourceManager=new Eu}findIOHandler(){const e=this.modelUrl;if(e.load!=null)this.handler=e;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{const t=this.io.getLoadHandlers(e,this.loadOptions);if(t.length===0)t.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(t.length>1)throw new Error(`Found more than one (${t.length}) load handlers for URL '${[e]}'`);this.handler=t[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const e=this.handler.load();return ye(e)?e.then(t=>t.getWeightStream==null?this.loadSync(t):this.loadStreaming(t)):this.loadSync(e)}loadSync(e){const t=this.io.decodeWeights(e.weightData,e.weightSpecs);return this.loadWithWeightMap(e,t)}async loadStreaming(e){if(e.getWeightStream==null)throw new Error("Model artifacts missing streamWeights function");const t=await bt(e.getWeightStream(),e.weightSpecs);return this.loadWithWeightMap(e,t)}loadWithWeightMap(e,t){this.artifacts=e;const s=this.artifacts.modelTopology;let r=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const i=this.artifacts.userDefinedMetadata;i.signature!=null&&(r=i.signature),i.structuredOutputKeys!=null&&(this.structuredOutputKeys=i.structuredOutputKeys)}if(this.signature=r,this.version=`${s.versions.producer}.${s.versions.minConsumer}`,this.executor=new ge(Qe.Instance.transformGraph(s,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(t),this.executor.resourceManager=this.resourceManager,e.modelInitializer!=null&&e.modelInitializer.node!=null){const i=Qe.Instance.transformGraph(e.modelInitializer);this.initializer=new ge(i),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=e.initializerSignature}return!0}async save(e,t){if(typeof e=="string"){const s=this.io.getSaveHandlers(e);if(s.length===0)throw new Error(`Cannot find any save handlers for URL '${e}'`);if(s.length>1)throw new Error(`Found more than one (${s.length}) save handlers for URL '${e}'`);e=s[0]}if(e.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}addStructuredOutputNames(e){if(this.structuredOutputKeys){const t=e instanceof te?[e]:e,s={};return t.forEach((r,i)=>s[this.structuredOutputKeys[i]]=r),s}return e}predict(e,t){const s=this.execute(e,this.outputNodes);return this.addStructuredOutputNames(s)}async predictAsync(e,t){const s=await this.executeAsync(e,this.outputNodes);return this.addStructuredOutputNames(s)}normalizeInputs(e){var t;if(!(e instanceof te)&&!Array.isArray(e)){const i=(t=this.signature)===null||t===void 0?void 0:t.inputs;if(i!=null)for(const u in i){const o=i[u];o.resourceId!=null&&(e[u]=this.resourceIdToCapturedInput[o.resourceId])}return e}e=Array.isArray(e)?e:[e];const s=Object.keys(this.resourceIdToCapturedInput).length;if(e.length+s!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-s} non-resource placeholders, while there are ${e.length} input tensors provided.`);let r=0;return this.inputNodes.reduce((i,u)=>{var o,p,l;const m=(l=(p=(o=this.signature)===null||o===void 0?void 0:o.inputs)===null||p===void 0?void 0:p[u])===null||l===void 0?void 0:l.resourceId;return m!=null?i[u]=this.resourceIdToCapturedInput[m]:i[u]=e[r++],i},{})}normalizeOutputs(e){return e=e||this.outputNodes,Array.isArray(e)?e:[e]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(e){if(this.resourceIdToCapturedInput={},this.initializerSignature){const t=this.initializerSignature.outputs,s=Object.keys(t);for(let r=0;r<s.length;r++){const i=s[r],u=t[i];this.resourceIdToCapturedInput[u.resourceId]=e[r]}}}execute(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const s=this.executor.execute(e,t);return s.length>1?s:s[0]}async executeAsync(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const s=await this.executor.executeAsync(e,t);return s.length>1?s:s[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((t,s)=>(t[s]=[e[s]],t),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&wn(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function st(a,e={},t=Ot){if(a==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");e==null&&(e={}),e.fromTFHub&&typeof a=="string"&&(a=zu(a));const s=new Du(a,e,t);return await s.load(),s}function zu(a){return a.endsWith("/")||(a=a+"/"),`${a}${ju}${$u}`}const xu="https://www.kaggle.com/models/google/inception-v3/TfJs/classification/2",rt="indexeddb://inception-v3",pe=299,Lu=100,Et=2,Cu=["background","tench","goldfish","great white shark","tiger shark","hammerhead","electric ray","stingray","cock","hen","ostrich","brambling","goldfinch","house finch","junco","indigo bunting","robin","bulbul","jay","magpie","chickadee","water ouzel","kite","bald eagle","vulture","great grey owl","European fire salamander","common newt","eft","spotted salamander","axolotl","bullfrog","tree frog","tailed frog","loggerhead","leatherback turtle","mud turtle","terrapin","box turtle","banded gecko","common iguana","American chameleon","whiptail","agama","frilled lizard","alligator lizard","Gila monster","green lizard","African chameleon","Komodo dragon","African crocodile","American alligator","triceratops","thunder snake","ringneck snake","hognose snake","green snake","king snake","garter snake","water snake","vine snake","night snake","boa constrictor","rock python","Indian cobra","green mamba","sea snake","horned viper","diamondback","sidewinder","trilobite","harvestman","scorpion","black and gold garden spider","barn spider","garden spider","black widow","tarantula","wolf spider","tick","centipede","black grouse","ptarmigan","ruffed grouse","prairie chicken","peacock","quail","partridge","African grey","macaw","sulphur-crested cockatoo","lorikeet","coucal","bee eater","hornbill","hummingbird","jacamar","toucan","drake","red-breasted merganser","goose","black swan","tusker","echidna","platypus","wallaby","koala","wombat","jellyfish","sea anemone","brain coral","flatworm","nematode","conch","snail","slug","sea slug","chiton","chambered nautilus","Dungeness crab","rock crab","fiddler crab","king crab","American lobster","spiny lobster","crayfish","hermit crab","isopod","white stork","black stork","spoonbill","flamingo","little blue heron","American egret","bittern","crane","limpkin","European gallinule","American coot","bustard","ruddy turnstone","red-backed sandpiper","redshank","dowitcher","oystercatcher","pelican","king penguin","albatross","grey whale","killer whale","dugong","sea lion","Chihuahua","Japanese spaniel","Maltese dog","Pekinese","Shih-Tzu","Blenheim spaniel","papillon","toy terrier","Rhodesian ridgeback","Afghan hound","basset","beagle","bloodhound","bluetick","black-and-tan coonhound","Walker hound","English foxhound","redbone","borzoi","Irish wolfhound","Italian greyhound","whippet","Ibizan hound","Norwegian elkhound","otterhound","Saluki","Scottish deerhound","Weimaraner","Staffordshire bullterrier","American Staffordshire terrier","Bedlington terrier","Border terrier","Kerry blue terrier","Irish terrier","Norfolk terrier","Norwich terrier","Yorkshire terrier","wire-haired fox terrier","Lakeland terrier","Sealyham terrier","Airedale","cairn","Australian terrier","Dandie Dinmont","Boston bull","miniature schnauzer","giant schnauzer","standard schnauzer","Scotch terrier","Tibetan terrier","silky terrier","soft-coated wheaten terrier","West Highland white terrier","Lhasa","flat-coated retriever","curly-coated retriever","golden retriever","Labrador retriever","Chesapeake Bay retriever","German short-haired pointer","vizsla","English setter","Irish setter","Gordon setter","Brittany spaniel","clumber","English springer","Welsh springer spaniel","cocker spaniel","Sussex spaniel","Irish water spaniel","kuvasz","schipperke","groenendael","malinois","briard","kelpie","komondor","Old English sheepdog","Shetland sheepdog","collie","Border collie","Bouvier des Flandres","Rottweiler","German shepherd","Doberman","miniature pinscher","Greater Swiss Mountain dog","Bernese mountain dog","Appenzeller","EntleBucher","boxer","bull mastiff","Tibetan mastiff","French bulldog","Great Dane","Saint Bernard","Eskimo dog","malamute","Siberian husky","dalmatian","affenpinscher","basenji","pug","Leonberg","Newfoundland","Great Pyrenees","Samoyed","Pomeranian","chow","keeshond","Brabancon griffon","Pembroke","Cardigan","toy poodle","miniature poodle","standard poodle","Mexican hairless","timber wolf","white wolf","red wolf","coyote","dingo","dhole","African hunting dog","hyena","red fox","kit fox","Arctic fox","grey fox","tabby","tiger cat","Persian cat","Siamese cat","Egyptian cat","cougar","lynx","leopard","snow leopard","jaguar","lion","tiger","cheetah","brown bear","American black bear","ice bear","sloth bear","mongoose","meerkat","tiger beetle","ladybug","ground beetle","long-horned beetle","leaf beetle","dung beetle","rhinoceros beetle","weevil","fly","bee","ant","grasshopper","cricket","walking stick","cockroach","mantis","cicada","leafhopper","lacewing","dragonfly","damselfly","admiral","ringlet","monarch","cabbage butterfly","sulphur butterfly","lycaenid","starfish","sea urchin","sea cucumber","wood rabbit","hare","Angora","hamster","porcupine","fox squirrel","marmot","beaver","guinea pig","sorrel","zebra","hog","wild boar","warthog","hippopotamus","ox","water buffalo","bison","ram","bighorn","ibex","hartebeest","impala","gazelle","Arabian camel","llama","weasel","mink","polecat","black-footed ferret","otter","skunk","badger","armadillo","three-toed sloth","orangutan","gorilla","chimpanzee","gibbon","siamang","guenon","patas","baboon","macaque","langur","colobus","proboscis monkey","marmoset","capuchin","howler monkey","titi","spider monkey","squirrel monkey","Madagascar cat","indri","Indian elephant","African elephant","lesser panda","giant panda","barracouta","eel","coho","rock beauty","anemone fish","sturgeon","gar","lionfish","puffer","abacus","abaya","academic gown","accordion","acoustic guitar","aircraft carrier","airliner","airship","altar","ambulance","amphibian","analog clock","apiary","apron","ashcan","assault rifle","backpack","bakery","balance beam","balloon","ballpoint","Band Aid","banjo","bannister","barbell","barber chair","barbershop","barn","barometer","barrel","barrow","baseball","basketball","bassinet","bassoon","bathing cap","bath towel","bathtub","beach wagon","beacon","beaker","bearskin","beer bottle","beer glass","bell cote","bib","bicycle-built-for-two","bikini","binder","binoculars","birdhouse","boathouse","bobsled","bolo tie","bonnet","bookcase","bookshop","bottlecap","bow","bow tie","brass","brassiere","breakwater","breastplate","broom","bucket","buckle","bulletproof vest","bullet train","butcher shop","cab","caldron","candle","cannon","canoe","can opener","cardigan","car mirror","carousel","carpenter's kit","carton","car wheel","cash machine","cassette","cassette player","castle","catamaran","CD player","cello","cellular telephone","chain","chainlink fence","chain mail","chain saw","chest","chiffonier","chime","china cabinet","Christmas stocking","church","cinema","cleaver","cliff dwelling","cloak","clog","cocktail shaker","coffee mug","coffeepot","coil","combination lock","computer keyboard","confectionery","container ship","convertible","corkscrew","cornet","cowboy boot","cowboy hat","cradle","crane","crash helmet","crate","crib","Crock Pot","croquet ball","crutch","cuirass","dam","desk","desktop computer","dial telephone","diaper","digital clock","digital watch","dining table","dishrag","dishwasher","disk brake","dock","dogsled","dome","doormat","drilling platform","drum","drumstick","dumbbell","Dutch oven","electric fan","electric guitar","electric locomotive","entertainment center","envelope","espresso maker","face powder","feather boa","file","fireboat","fire engine","fire screen","flagpole","flute","folding chair","football helmet","forklift","fountain","fountain pen","four-poster","freight car","French horn","frying pan","fur coat","garbage truck","gasmask","gas pump","goblet","go-kart","golf ball","golfcart","gondola","gong","gown","grand piano","greenhouse","grille","grocery store","guillotine","hair slide","hair spray","half track","hammer","hamper","hand blower","hand-held computer","handkerchief","hard disc","harmonica","harp","harvester","hatchet","holster","home theater","honeycomb","hook","hoopskirt","horizontal bar","horse cart","hourglass","iPod","iron","jack-o'-lantern","jean","jeep","jersey","jigsaw puzzle","jinrikisha","joystick","kimono","knee pad","knot","lab coat","ladle","lampshade","laptop","lawn mower","lens cap","letter opener","library","lifeboat","lighter","limousine","liner","lipstick","Loafer","lotion","loudspeaker","loupe","lumbermill","magnetic compass","mailbag","mailbox","maillot","maillot","manhole cover","maraca","marimba","mask","matchstick","maypole","maze","measuring cup","medicine chest","megalith","microphone","microwave","military uniform","milk can","minibus","miniskirt","minivan","missile","mitten","mixing bowl","mobile home","Model T","modem","monastery","monitor","moped","mortar","mortarboard","mosque","mosquito net","motor scooter","mountain bike","mountain tent","mouse","mousetrap","moving van","muzzle","nail","neck brace","necklace","nipple","notebook","obelisk","oboe","ocarina","odometer","oil filter","organ","oscilloscope","overskirt","oxcart","oxygen mask","packet","paddle","paddlewheel","padlock","paintbrush","pajama","palace","panpipe","paper towel","parachute","parallel bars","park bench","parking meter","passenger car","patio","pay-phone","pedestal","pencil box","pencil sharpener","perfume","Petri dish","photocopier","pick","pickelhaube","picket fence","pickup","pier","piggy bank","pill bottle","pillow","ping-pong ball","pinwheel","pirate","pitcher","plane","planetarium","plastic bag","plate rack","plow","plunger","Polaroid camera","pole","police van","poncho","pool table","pop bottle","pot","potter's wheel","power drill","prayer rug","printer","prison","projectile","projector","puck","punching bag","purse","quill","quilt","racer","racket","radiator","radio","radio telescope","rain barrel","recreational vehicle","reel","reflex camera","refrigerator","remote control","restaurant","revolver","rifle","rocking chair","rotisserie","rubber eraser","rugby ball","rule","running shoe","safe","safety pin","saltshaker","sandal","sarong","sax","scabbard","scale","school bus","schooner","scoreboard","screen","screw","screwdriver","seat belt","sewing machine","shield","shoe shop","shoji","shopping basket","shopping cart","shovel","shower cap","shower curtain","ski","ski mask","sleeping bag","slide rule","sliding door","slot","snorkel","snowmobile","snowplow","soap dispenser","soccer ball","sock","solar dish","sombrero","soup bowl","space bar","space heater","space shuttle","spatula","speedboat","spider web","spindle","sports car","spotlight","stage","steam locomotive","steel arch bridge","steel drum","stethoscope","stole","stone wall","stopwatch","stove","strainer","streetcar","stretcher","studio couch","stupa","submarine","suit","sundial","sunglass","sunglasses","sunscreen","suspension bridge","swab","sweatshirt","swimming trunks","swing","switch","syringe","table lamp","tank","tape player","teapot","teddy","television","tennis ball","thatch","theater curtain","thimble","thresher","throne","tile roof","toaster","tobacco shop","toilet seat","torch","totem pole","tow truck","toyshop","tractor","trailer truck","tray","trench coat","tricycle","trimaran","tripod","triumphal arch","trolleybus","trombone","tub","turnstile","typewriter keyboard","umbrella","unicycle","upright","vacuum","vase","vault","velvet","vending machine","vestment","viaduct","violin","volleyball","waffle iron","wall clock","wallet","wardrobe","warplane","washbasin","washer","water bottle","water jug","water tower","whiskey jug","whistle","wig","window screen","window shade","Windsor tie","wine bottle","wing","wok","wooden spoon","wool","worm fence","wreck","yawl","yurt","web site","comic book","crossword puzzle","street sign","traffic light","book jacket","menu","plate","guacamole","consomme","hot pot","trifle","ice cream","ice lolly","French loaf","bagel","pretzel","cheeseburger","hotdog","mashed potato","head cabbage","broccoli","cauliflower","zucchini","spaghetti squash","acorn squash","butternut squash","cucumber","artichoke","bell pepper","cardoon","mushroom","Granny Smith","strawberry","orange","lemon","fig","pineapple","banana","jackfruit","custard apple","pomegranate","hay","carbonara","chocolate sauce","dough","meat loaf","pizza","potpie","burrito","red wine","espresso","cup","eggnog","alp","bubble","cliff","coral reef","geyser","lakeside","promontory","sandbar","seashore","valley","volcano","ballplayer","groom","scuba diver","rapeseed","daisy","yellow lady's slipper","corn","acorn","hip","buckeye","coral fungus","agaric","gyromitra","stinkhorn","earthstar","hen-of-the-woods","bolete","ear","toilet tissue"];function Vu({img:a,model:e,onSuccess:t}){return V(()=>{const s=vn(a),r=wt.resizeBilinear(s,[pe,pe],!0).div(255).reshape([1,pe,pe,3]),i=e.predict(r);if(Array.isArray(i)||!(i instanceof te))throw new Error("Something went wrong. Unexpected result type");const{indices:u,values:o}=Tt(i,Lu),p=o.asType("int32").dataSync(),l=Array.from(u.dataSync());t(l.reduce((m,c,d)=>p[d]<=Et?m:[...m,{label:Cu[c]??"Unknown",confidence:p[d]}],[]))})}function Pu({className:a,data:e,...t}){return e.length?h.jsx("div",{className:de("overflow-x-auto rounded-box border border-base-content/5 bg-base-100 p-1 max-h-60",a),children:h.jsxs("table",{className:"table w-full table-pin-rows",...t,children:[h.jsx("thead",{children:h.jsxs("tr",{children:[h.jsx("th",{children:"Label"}),h.jsx("th",{children:"Probability"})]})}),h.jsx("tbody",{children:e.map(({label:s,confidence:r})=>h.jsxs("tr",{children:[h.jsx("td",{className:"capitalize",children:s}),h.jsx("td",{className:de(r<=Et&&"text-error"),children:r})]},s))})]})}):null}function Fu({className:a,...e}){return h.jsxs("section",{className:de("collapse bg-base-100 border-base-300 border collapse-arrow",a),...e,children:[h.jsx("input",{type:"checkbox"}),h.jsx("div",{className:"collapse-title font-semibold",children:"How to achieve precise predictions"}),h.jsxs("div",{className:"collapse-content",children:[h.jsx("h6",{children:"1️⃣ Number of objects per image"}),h.jsxs("ul",{children:[h.jsx("li",{children:"Single object per image → Best results."}),h.jsx("ul",{children:h.jsx("li",{children:"Inception V3 is optimized for recognizing one primary object."})}),h.jsx("li",{children:"Multiple objects → Accuracy drops."}),h.jsxs("ul",{children:[h.jsx("li",{children:"The model may predict the most prominent object, or mix predictions from different objects."}),h.jsx("li",{children:"If you want multiple object recognition, consider object detection models (e.g., Faster R-CNN, YOLO, SSD) instead."})]})]}),h.jsx("h6",{children:"2️⃣ Object size and placement"}),h.jsxs("ul",{children:[h.jsx("li",{children:"Centered, large object → More precise predictions."}),h.jsx("li",{children:"Small or off-center object → Model might miss it or focus on background."}),h.jsx("li",{children:"Cropping the object before feeding it improves precision."})]}),h.jsx("h6",{children:"3️⃣ Background"}),h.jsxs("ul",{children:[h.jsx("li",{children:"Simple, uncluttered backgrounds → Better predictions."}),h.jsx("li",{children:"Busy or similar-colored background → Model can confuse background with object features."}),h.jsx("li",{children:"Pretrained Inception V3 isn’t trained to ignore backgrounds, so segmentation or cropping helps."})]}),h.jsx("h6",{children:"4️⃣ Image quality"}),h.jsxs("ul",{children:[h.jsx("li",{children:"High resolution, sharp images → Higher accuracy."}),h.jsx("li",{children:"Blur, noise, compression artifacts → Reduces confidence and may misclassify."}),h.jsx("li",{children:"Consistent lighting helps because Inception V3 isn’t robust to extreme lighting changes."})]}),h.jsx("h6",{children:"5️⃣ Object viewpoint and orientation"}),h.jsxs("ul",{children:[h.jsx("li",{children:"Canonical views (like ImageNet standard: front, side, or typical pose) → Better."}),h.jsx("li",{children:"Extreme rotations, unusual angles, occlusions → Accuracy drops."})]}),h.jsx("h6",{children:"⚡ Summary (Photo-wise)"}),h.jsx("table",{className:"table",children:h.jsxs("tbody",{children:[h.jsxs("tr",{children:[h.jsx("th",{children:"Factor"}),h.jsx("th",{children:"Ideal for Inception V3"})]}),h.jsxs("tr",{children:[h.jsx("td",{children:"Objects per image"}),h.jsx("td",{children:"Single"})]}),h.jsxs("tr",{children:[h.jsx("td",{children:"Object placement"}),h.jsx("td",{children:"Centered"})]}),h.jsxs("tr",{children:[h.jsx("td",{children:"Object size"}),h.jsx("td",{children:"Large relative to frame"})]}),h.jsxs("tr",{children:[h.jsx("td",{children:"Background"}),h.jsx("td",{children:"Simple / clean"})]}),h.jsxs("tr",{children:[h.jsx("td",{children:"Image quality"}),h.jsx("td",{children:"Sharp, good lighting"})]}),h.jsxs("tr",{children:[h.jsx("td",{children:"Object viewpoint"}),h.jsx("td",{children:"Standard angles"})]})]})})]})]})}async function Ru(){await Tn();try{return await st(rt)}catch{const a=await st(xu,{fromTFHub:!0});return await a.save(rt),a}}function Bu(){return zt({queryKey:["inception-v3-model"],queryFn:async()=>await Ru(),staleTime:1/0,gcTime:1/0})}function qu(){const[a,e]=Te.useState([]),[t,s]=Te.useState(!1),[r,i]=Te.useState(null),u=jt(r),{isLoading:o,data:p,error:l}=Bu();return h.jsx("section",{className:de("prose p-4",(o||!!l)&&"text-center"),children:o?h.jsxs(h.Fragment,{children:[h.jsx("p",{children:"Wait before model data will load..."}),h.jsx("p",{className:"loading loading-bars loading-xl"})]}):h.jsxs(h.Fragment,{children:[!!l&&h.jsxs(h.Fragment,{children:[h.jsx("h4",{children:"Oops, something went wrong"}),h.jsx("p",{className:"text-error",children:l.message}),h.jsx("p",{children:"Try to refresh the page"})]}),!l&&!!p&&h.jsxs(h.Fragment,{children:[h.jsx("h2",{children:"Inception V3 - neural network architecture for image classification"}),h.jsx("input",{type:"file",onChange:m=>{i(m.target.files?.[0]??null),s(!!m.target.files?.[0]),e([])},className:"file-input"}),t&&h.jsxs("div",{className:"text-center",children:[h.jsx("p",{children:"Wait before computation will finish..."}),h.jsx("p",{className:"loading loading-spinner loading-xl"})]}),h.jsx(Pu,{data:a}),u?h.jsx("img",{src:u,alt:"Loaded image",onLoad:m=>{Vu({model:p,onSuccess:c=>{e(c),s(!1)},img:m.currentTarget})}}):h.jsx("p",{children:"Please select an image first"}),h.jsx(Dt,{}),h.jsx(Fu,{})]})]})})}function Qu(){return h.jsx(xt,{children:h.jsx(qu,{})})}export{Qu as default};
