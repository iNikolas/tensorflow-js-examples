import * as tf from "@tensorflow/tfjs";

import { batchSize, dataSize, learningRate, stepSize } from "../config";

export function splitIntoChunks(num: number, size = 2) {
  const str = String(Math.floor(num));
  const chunks = [];
  for (let i = 0; i < str.length; i += size) {
    chunks.push(Number(str.slice(i, i + size)));
  }
  return chunks;
}

export async function loadModel(args: tf.ModelFitArgs) {
  const jsxs = [];
  const jsys = [];

  for (let i = 0; i < dataSize; i = i + stepSize) {
    jsxs.push(i);
    jsys.push(i * i);
  }

  await tf.ready();

  const xs = tf.tensor1d(jsxs);
  const ys = tf.tensor1d(jsys);

  const model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [1], units: 20, activation: "relu" })
  );

  model.add(tf.layers.dense({ units: 20, activation: "relu" }));

  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: tf.losses.meanSquaredError,
  });

  const history = await model.fit(xs, ys, {
    batchSize,
    ...args,
  });

  tf.dispose([xs, ys]);

  return { model, history };
}
