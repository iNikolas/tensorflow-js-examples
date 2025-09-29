import{db as d,dc as x}from"./register_all_kernels-YBgeH-G1.js";import{j as s}from"./index-qOZqXyW5.js";import{c,s as m,f as u}from"./helpers-Cs5YJYh0.js";/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function g(e){return new d(e)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function b(e){return new x(e)}function p({className:e,value:r,children:t,decimals:a=2,...i}){const o=Number.parseInt(Number.parseFloat(r.toFixed(a)).toString().split(".")[1]);return s.jsxs("p",{className:c("font-mono text-2xl",e),...i,children:[s.jsxs("strong",{children:[t," "]}),s.jsxs("span",{className:"countdown",children:[m(r).map((n,l)=>s.jsx("span",{style:{"--value":n},"aria-live":"polite","aria-label":n.toString(),children:n},`loss-${l}`)),!!a&&!Number.isNaN(o)&&s.jsxs(s.Fragment,{children:[".",m(o).map((n,l)=>s.jsx("span",{style:{"--value":n},"aria-live":"polite","aria-label":n.toString(),children:n},`loss-${l}`))]})]})]})}function h({loss:e,accuracy:r,trainingProgress:t,className:a,...i}){return s.jsxs("section",{className:c("prose text-center",a),...i,children:[s.jsxs("p",{children:["Training model... (",u(t)," %)"]}),s.jsx("progress",{className:"progress progress-primary w-56",value:t,max:"100"}),e!==1/0&&s.jsx(p,{decimals:4,value:e,children:"Loss:"}),r!=null&&r!==1/0&&s.jsx(p,{decimals:1,value:r*100,children:"Accuracy(%):"})]})}export{h as T,b as d,g as s};
