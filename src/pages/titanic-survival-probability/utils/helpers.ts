import {
  concat,
  LabelEncoder,
  readCSV,
  getDummies,
  MinMaxScaler,
} from "danfojs";
import * as tf from "@tensorflow/tfjs";
import type { DataFrame, Series } from "danfojs";
import type {
  ArrayType1D,
  ArrayType2D,
  CsvInputOptionsBrowser,
} from "node_modules/danfojs/dist/danfojs-base/shared/types";

import type { Limits, Profile } from "../types";
import testCSV from "../assets/datasets/test.csv?url";
import trainCSV from "../assets/datasets/train.csv?url";
import { ageGroups, datasetColumns, layers, validationSplit } from "../config";

function ageToBucket(age: number) {
  return ageGroups.findIndex(({ max }) => age <= max);
}

function normalizeDataset(dataFrame: DataFrame): {
  scaled: DataFrame;
  scaler: MinMaxScaler;
} {
  const scaler = new MinMaxScaler();

  const scaled = scaler.fitTransform(dataFrame);

  return { scaled, scaler };
}

function bucketColumn(
  dataFrame: DataFrame,
  {
    column,
    callback,
    drop = true,
  }: { column: string; callback: (x: number) => void; drop?: boolean }
) {
  const buckets = dataFrame[column].apply(callback);

  dataFrame.addColumn(`${column}_bucket`, buckets, { inplace: true });

  if (drop) {
    dataFrame.drop({ columns: [column], inplace: true });
  }

  return dataFrame;
}

function oneHotEncoding(
  dataFrame: DataFrame,
  { column, columns }: { column: string; columns: [string, string] }
): DataFrame {
  const oneHotEncode = getDummies(dataFrame[column]);

  dataFrame.drop({ columns: [column], inplace: true });

  columns.forEach((columnName, index) => {
    dataFrame.addColumn(columnName, oneHotEncode[oneHotEncode.columns[index]], {
      inplace: true,
    });
  });

  return dataFrame;
}

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

  return { dataFrame, classes: encode.classes };
}

async function prepareData(): Promise<{
  x: tf.Tensor2D;
  y: tf.Tensor1D;
  scaler: MinMaxScaler;
  columns: string[];
  sample: number[];
  embarkedClasses: {
    [key: string]: number;
  };
  referenceData: { columns: string[]; data: ArrayType1D | ArrayType2D };
}> {
  const allData = await combineData([trainCSV, testCSV]);

  const onlyFull = dropEmptyRows(
    removeFeatureColumns(allData, ["Name", "PassengerId", "Ticket", "Cabin"])
  ).resetIndex();

  const fullData = onlyFull.values;
  const fullDataColumns = onlyFull.columns;

  const { dataFrame: emberkedEncodedDataFrame, classes: embarkedClasses } =
    encodeStringColumns({
      dataFrame: onlyFull,
      column: "Embarked",
    });

  const data = bucketColumn(
    oneHotEncoding(
      encodeStringColumns({
        dataFrame: emberkedEncodedDataFrame,
        column: "Sex",
      }).dataFrame,
      { column: "Sex", columns: ["male", "female"] }
    ),
    { column: "Age", callback: ageToBucket }
  );

  const dataArray: number[][] = await data.tensor.array();

  const { scaled, scaler } = normalizeDataset(data);

  const sample = dataArray[Math.floor(Math.random() * dataArray.length)];

  return {
    x: scaled.iloc({ columns: ["1:"] }).tensor,
    y: scaled["Survived"].tensor,
    scaler,
    columns: scaled.columns,
    sample,
    embarkedClasses,
    referenceData: {
      columns: fullDataColumns,
      data: fullData,
    },
  };
}

export async function loadModel({
  learningRate,
  callbacks,
  ...args
}: Omit<tf.ModelFitArgs, "callbacks"> & {
  learningRate: number;
  callbacks?: tf.CustomCallbackArgs;
}) {
  const { x, y, scaler, columns, sample, embarkedClasses, referenceData } =
    await prepareData();

  await tf.ready();

  const model = tf.sequential();

  layers.forEach((layer, index) =>
    model.add(
      tf.layers.dense({
        ...layer,
        ...(index === 0 && { inputShape: [x.shape[1]] }),
      })
    )
  );

  model.compile({
    optimizer: tf.train.momentum(learningRate, 0.9, true),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  const history = await model.fit(x, y, {
    validationSplit,
    shuffle: true,
    callbacks,
    ...args,
  });

  tf.dispose([x, y]);

  return {
    model,
    history,
    scaler,
    columns,
    sample,
    embarkedClasses,
    referenceData,
  };
}

function getColumnIndices(columns: string[]) {
  const siblingsAmountIndex = columns.indexOf(datasetColumns.siblingsAmount);
  const familyAmountIndex = columns.indexOf(datasetColumns.familyAmount);
  const fareIndex = columns.indexOf(datasetColumns.fare);
  const maleIndex = columns.indexOf(datasetColumns.male);
  const ageBucketIndex = columns.indexOf(datasetColumns.ageBucket);
  const passengerClassIndex = columns.indexOf(datasetColumns.passengerClass);
  const portIndex = columns.indexOf(datasetColumns.port);
  const femaleIndex = columns.indexOf(datasetColumns.female);

  return {
    siblingsAmountIndex,
    familyAmountIndex,
    fareIndex,
    maleIndex,
    ageBucketIndex,
    passengerClassIndex,
    portIndex,
    femaleIndex,
  };
}

export function calculateLimits({
  scaler,
  columns,
}: {
  scaler: MinMaxScaler;
  columns: string[];
}): Limits {
  const maxLimits = scaler.inverseTransform(
    Array.from({ length: columns.length }).fill(1)
  );

  const minLimits = scaler.inverseTransform(
    Array.from({ length: columns.length }).fill(0)
  );

  const { siblingsAmountIndex, familyAmountIndex, fareIndex } =
    getColumnIndices(columns);

  return {
    min: {
      siblingsAmount: minLimits[siblingsAmountIndex],
      familyAmount: minLimits[familyAmountIndex],
      fare: minLimits[fareIndex],
    },
    max: {
      siblingsAmount: maxLimits[siblingsAmountIndex],
      familyAmount: maxLimits[familyAmountIndex],
      fare: maxLimits[fareIndex],
    },
  };
}

export function sampleToProfile({
  sample,
  columns,
  embarkedClasses,
}: {
  sample: number[];
  columns: string[];
  embarkedClasses: { [key: string]: number };
}): Profile {
  const {
    siblingsAmountIndex,
    familyAmountIndex,
    fareIndex,
    maleIndex,
    ageBucketIndex,
    passengerClassIndex,
    portIndex,
  } = getColumnIndices(columns);

  const port = Object.entries(embarkedClasses).find(
    ([, value]) => sample[portIndex] === value
  )?.[0];

  return {
    male: sample[maleIndex] === 0 ? 0 : 1,
    age:
      ageGroups[sample[ageBucketIndex]].max < Infinity
        ? Math.floor(ageGroups[sample[ageBucketIndex]].max * 0.7)
        : 50,
    passengerClass: sample[passengerClassIndex],
    siblingsAmount: sample[siblingsAmountIndex],
    familyAmount: sample[familyAmountIndex],
    fare: sample[fareIndex],
    port: port ?? "C",
  };
}

function assertIsNumericArray(array: unknown): asserts array is number[] {
  if (!Array.isArray(array) || array.some((item) => typeof item !== "number")) {
    throw new Error("Expected numeric array");
  }
}

export function profileToSample({
  profile,
  columns,
  embarkedClasses,
}: {
  profile: Profile;
  columns: string[];
  embarkedClasses: { [key: string]: number };
}): number[] {
  const result = Array.from({ length: columns.length }).fill(0);

  assertIsNumericArray(result);

  const {
    siblingsAmountIndex,
    familyAmountIndex,
    fareIndex,
    maleIndex,
    ageBucketIndex,
    passengerClassIndex,
    portIndex,
    femaleIndex,
  } = getColumnIndices(columns);

  result[siblingsAmountIndex] = profile.siblingsAmount;
  result[familyAmountIndex] = profile.familyAmount;
  result[fareIndex] = profile.fare;
  result[maleIndex] = Number(profile.male);
  result[femaleIndex] = 1 - profile.male;
  result[ageBucketIndex] = ageGroups.findIndex(({ max }) => profile.age <= max);
  result[passengerClassIndex] = Number(profile.passengerClass);
  result[portIndex] = embarkedClasses[profile.port];

  return result;
}
