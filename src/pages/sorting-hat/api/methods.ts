import * as tf from "@tensorflow/tfjs";
import classes from "public/model/sorting-hat/classes.json";

import { localKey, modelPath } from "../config";

export async function loadModel() {
  await tf.ready();

  try {
    const cachedModel = await tf.loadLayersModel(localKey);
    return { model: cachedModel, classes };
  } catch {
    const model = await tf.loadLayersModel(modelPath);
    await model.save(localKey);

    return { model, classes };
  }
}
