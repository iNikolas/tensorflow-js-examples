import * as tf from "@tensorflow/tfjs";

import {
  expectedImageSize,
  maxPredictionsPerImage,
  modelLabels,
  thresholdConfidence,
} from "../config";
import type { Model, PredictionResult } from "../types";

export function predict({
  img,
  model,
  onSuccess,
}: {
  img: HTMLImageElement;
  model: Model;
  onSuccess: (data: PredictionResult[]) => void;
}) {
  return tf.tidy(() => {
    const imageTensor = tf.browser.fromPixels(img);

    const readyfied = tf.image
      .resizeBilinear(imageTensor, [expectedImageSize, expectedImageSize], true)
      .div(255)
      .reshape([1, expectedImageSize, expectedImageSize, 3]);

    const result = model.predict(readyfied);

    if (Array.isArray(result) || !(result instanceof tf.Tensor)) {
      throw new Error("Something went wrong. Unexpected result type");
    }

    const { indices, values } = tf.topk(result, maxPredictionsPerImage);

    const probabilities = values.asType("int32").dataSync<"int32">();
    const indicesArray = Array.from(indices.dataSync<"int32">());

    onSuccess(
      indicesArray.reduce<PredictionResult[]>((acc, val, index) => {
        const confidence = probabilities[index];

        if (confidence <= thresholdConfidence) {
          return acc;
        }

        return [
          ...acc,
          {
            label: modelLabels[val] ?? "Unknown",
            confidence: probabilities[index],
          },
        ];
      }, [])
    );
  });
}
