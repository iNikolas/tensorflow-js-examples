import React from "react";

import type { PredictionResult } from "../../types";

export interface PredictionDialogProps
  extends React.DialogHTMLAttributes<HTMLDialogElement> {
  predictions: PredictionResult | null;
  deathEaterThreshold?: number;
}
