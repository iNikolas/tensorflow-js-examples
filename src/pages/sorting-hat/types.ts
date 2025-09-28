import type { loadModel } from "./api";
import type { predictions } from "./config";

export type Model = Awaited<ReturnType<typeof loadModel>>;

export type Prediction = (typeof predictions)[keyof typeof predictions];

export type PredictionResult = {
  [key in Prediction]: number;
};
