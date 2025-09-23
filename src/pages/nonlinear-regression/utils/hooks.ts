import React from "react";
import * as tf from "@tensorflow/tfjs";

import { loadModel } from "./helpers";
import { epochs, uiUpdateIntervalMs } from "../config";

export function useModel() {
  const [trainingProgress, setTrainingProgress] = React.useState(0);
  const [loss, setLoss] = React.useState(Infinity);

  const updatedTimestampRef = React.useRef(0);

  const [model, setModel] = React.useState<tf.Sequential | null>(null);

  const [isTraining, setIsTraining] = React.useState(false);

  React.useEffect(() => {
    const trainModel = async () => {
      if (model || isTraining) {
        return;
      }

      setIsTraining(true);

      const { model: data } = await loadModel({
        epochs,
        callbacks: {
          onEpochEnd: (epoch, log) => {
            setTrainingProgress((epoch / epochs) * 100);

            const currentTimestamp = Date.now();

            if (
              currentTimestamp - updatedTimestampRef.current >
              uiUpdateIntervalMs
            ) {
              updatedTimestampRef.current = currentTimestamp;

              setLoss(log?.loss ?? Infinity);
            }
          },
        },
      });

      setModel(data);

      setIsTraining(false);
    };

    trainModel();

    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, [isTraining, model]);

  return { trainingProgress, loss, model, isTraining };
}

export function usePrediction({
  value,
  model,
}: {
  value?: number;
  model?: tf.Sequential | null;
}) {
  const [prediction, setPrediction] = React.useState(0);

  const [isPredicting, setIsPredicting] = React.useState(false);

  React.useEffect(() => {
    const predict = async () => {
      if (!model || value == null) {
        return;
      }

      try {
        setIsPredicting(true);

        const prediction = tf.tidy(() => model.predict(tf.tensor([value])));

        if (Array.isArray(prediction)) {
          throw new Error("Something went wrong. Unexpected result type");
        }

        const predictionValue = Array.from(
          await prediction.data<"float32">()
        )[0];

        prediction.dispose();

        setPrediction(predictionValue);
      } finally {
        setIsPredicting(false);
      }
    };

    predict();
  }, [model, value]);

  return { prediction, isPredicting };
}
