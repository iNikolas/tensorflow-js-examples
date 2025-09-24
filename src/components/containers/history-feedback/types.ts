import React from "react";
import * as tf from "@tensorflow/tfjs";

export interface HistoryFeedbackProps
  extends React.HTMLAttributes<HTMLElement> {
  history: tf.History | null;
  threshold?: number;
}
