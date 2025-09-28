import { predictions } from "@/pages/sorting-hat/config";

import gryffindorCrest from "../assets/Gryffindor_crest.webp";
import hufflepuffCrest from "../assets/Hufflepuff_crest.webp";
import ravenclawCrest from "../assets/Ravenclaw_Crest_1.webp";
import slytherinCrest from "../assets/Slytherin_Crest.webp";

export const houses = [
  {
    name: predictions.gryffindor,
    crest: gryffindorCrest,
    color: "#a50000",
  },
  {
    name: predictions.hufflepuff,
    crest: hufflepuffCrest,
    color: "#ffb81c",
  },
  {
    name: predictions.ravenclaw,
    crest: ravenclawCrest,
    color: "#0e1a40",
  },
  {
    name: predictions.slytherin,
    crest: slytherinCrest,
    color: "#2a623d",
  },
];
