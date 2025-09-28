export const modelPath = `${
  import.meta.env.VITE_BASE_PATH ?? ""
}/model/sorting-hat/model.json`;

export const localKey = "indexeddb://sorting-hat";

export const predictions = {
  gryffindor: "gryffindor",
  hufflepuff: "hufflepuff",
  ravenclaw: "ravenclaw",
  slytherin: "slytherin",
  deatheater: "deatheater",
} as const;
