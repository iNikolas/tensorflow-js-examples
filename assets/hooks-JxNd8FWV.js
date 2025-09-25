import{R as r}from"./index-S5P_VTuv.js";function u(e){const[c,a]=r.useState("");return r.useEffect(()=>{a(t=>(t&&URL.revokeObjectURL(t),e?URL.createObjectURL(e):""))},[e]),c}export{u};
