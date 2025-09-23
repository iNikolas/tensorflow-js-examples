import * as tf from "@tensorflow/tfjs";
import type { DataFrame, Series } from "danfojs";
import { concat, LabelEncoder, readCSV } from "danfojs";
import type { CsvInputOptionsBrowser } from "node_modules/danfojs/dist/danfojs-base/shared/types";

import { batchSize, learningRate, trainingSplit } from "../config";
import testCSV from "../assets/datasets/test.csv?url";
import trainCSV from "../assets/datasets/train.csv?url";

function assertIsDataFrame(
  data: DataFrame | Series
): asserts data is DataFrame {
  if (data.$isSeries) {
    throw new Error("Expected DataFrame but got Series");
  }
}

function removeFeatureColumns(dataFrame: DataFrame, columns: string[]) {
  return dataFrame.drop({ columns });
}

function dropEmptyRows(dataFrame: DataFrame) {
  return dataFrame.dropNa();
}

async function readDataCSV(path: string, options?: CsvInputOptionsBrowser) {
  return await readCSV(path, options);
}

async function combineData(
  urlsList: string[],
  axis: 0 | 1 = 0
): Promise<DataFrame> {
  const dfList = await Promise.all(urlsList.map((url) => readDataCSV(url)));

  const result = concat({ dfList, axis });

  assertIsDataFrame(result);

  return result;
}

function encodeStringColumns({
  dataFrame,
  column,
}: {
  dataFrame: DataFrame;
  column: string;
}) {
  const encode = new LabelEncoder();

  encode.fit(dataFrame[column]);

  dataFrame[column] = encode.transform(dataFrame[column].values);

  return dataFrame;
}

async function prepareData() {
  const allData = await combineData([trainCSV, testCSV]);

  const onlyFull = dropEmptyRows(
    removeFeatureColumns(allData, ["Name", "PassengerId", "Ticket", "Cabin"])
  ).resetIndex();

  const data = encodeStringColumns({
    dataFrame: encodeStringColumns({ dataFrame: onlyFull, column: "Embarked" }),
    column: "Sex",
  });

  const trainDataAmount = Math.ceil(data.shape[0] * trainingSplit);
  const train = await data.sample(trainDataAmount);

  const test = data.drop({ index: train.index });

  return {
    train: {
      x: train.iloc({ columns: ["1:"] }).tensor,
      y: train["Survived"].tensor,
    },
    test: {
      x: test.iloc({ columns: ["1:"] }).tensor,
      y: test["Survived"].tensor,
    },
  };
}

export async function loadModel(args: tf.ModelFitArgs) {
  const { train, test } = await prepareData();

  await tf.ready();

  const xs = tf.tensor1d(train.x);
  const ys = tf.tensor1d(train.y);

  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [train.x.shape[1]],
      units: 120,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));

  model.add(tf.layers.dense({ units: 32, activation: "relu" }));

  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: tf.losses.sigmoidCrossEntropy,
    metrics: ["accuracy"],
  });

  const history = await model.fit(xs, ys, {
    batchSize,
    validationData: [test.x, test.y],
    ...args,
  });

  tf.dispose([xs, ys]);

  return { model, history };
}
