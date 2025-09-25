import{da as d,db as x}from"./register_all_kernels-CjX0k5XN.js";import{j as s}from"./index-Rp7MF6v_.js";import{c,s as m,f as u}from"./helpers-Cs5YJYh0.js";/**
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
 */function b(e){return new x(e)}function p({className:e,value:r,children:a,decimals:t=2,...i}){const o=Number.parseInt(Number.parseFloat(r.toFixed(t)).toString().split(".")[1]);return s.jsxs("p",{className:c("font-mono text-2xl",e),...i,children:[s.jsxs("strong",{children:[a," "]}),s.jsxs("span",{className:"countdown",children:[m(r).map((n,l)=>s.jsx("span",{style:{"--value":n},"aria-live":"polite","aria-label":n.toString(),children:n},`loss-${l}`)),!!t&&!Number.isNaN(o)&&s.jsxs(s.Fragment,{children:[".",m(o).map((n,l)=>s.jsx("span",{style:{"--value":n},"aria-live":"polite","aria-label":n.toString(),children:n},`loss-${l}`))]})]})]})}function h({loss:e,accuracy:r,trainingProgress:a,className:t,...i}){return s.jsxs("section",{className:c("prose text-center",t),...i,children:[s.jsxs("p",{children:["Training model... (",u(a)," %)"]}),s.jsx("progress",{className:"progress progress-primary w-56",value:a,max:"100"}),e!==1/0&&s.jsx(p,{decimals:4,value:e,children:"Loss:"}),r!=null&&r!==1/0&&s.jsx(p,{decimals:1,value:r*100,children:"Accuracy(%):"})]})}export{h as T,b as d,g as s};
