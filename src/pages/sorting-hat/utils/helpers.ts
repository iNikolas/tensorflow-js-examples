import * as tf from "@tensorflow/tfjs";

import type { Model, PredictionResult } from "../types";

function displayResults(
  predictions: Float32Array<ArrayBufferLike>
): PredictionResult {
  const ravenclaw = predictions[0] + predictions[2] + predictions[3];
  const gryffindor = predictions[1] + predictions[9];
  const hufflepuff = predictions[4] + predictions[8];
  const slytherin = predictions[6] + predictions[7];
  const deatheater = predictions[5];

  return {
    ravenclaw,
    gryffindor,
    hufflepuff,
    slytherin,
    deatheater,
  };
}

export function predict({
  img,
  model,
}: {
  img: HTMLImageElement;
  model: Model;
}) {
  return tf.tidy(() => {
    const imageTensor = tf.browser.fromPixels(img, 1);

    const expectedImageSize = model.model.layers[0].batchInputShape[1];

    if (expectedImageSize == null) {
      throw new Error("Something went wrong. Expected image size is null");
    }

    const readyfied = tf.image
      .resizeNearestNeighbor(
        imageTensor,
        [expectedImageSize, expectedImageSize],
        true
      )
      .div(255)
      .expandDims();

    const result = model.model.predict(readyfied);

    if (Array.isArray(result) || !(result instanceof tf.Tensor)) {
      throw new Error("Something went wrong. Unexpected result type");
    }

    const predictions = result.dataSync<"float32">();

    return displayResults(predictions);
  });
}
