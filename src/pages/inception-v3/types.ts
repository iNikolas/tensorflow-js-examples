import type { loadModel } from "./api";

export type Model = Awaited<ReturnType<typeof loadModel>>;

export interface PredictionResult {
  label: string;
  confidence: number;
}
