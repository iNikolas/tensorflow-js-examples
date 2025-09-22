import React from "react";
import * as tf from "@tensorflow/tfjs";

import { cn, formatNumber } from "@/utils/helpers";
import { MemoryUsage } from "@/components/containers/memory-usage";

async function loadModel({
  epochs = 300,
  callbacks,
}: {
  epochs?: number;
  callbacks?: tf.CustomCallbackArgs;
}) {
  const jsxs = [];
  const jsys = [];

  const dataSize = 10;
  const stepSize = 0.001;

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

  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: tf.losses.meanSquaredError,
  });

  const history = await model.fit(xs, ys, { epochs, callbacks });

  tf.dispose([xs, ys]);

  return { model, history };
}

export default function Page() {
  const [trainingProgress, setTrainingProgress] = React.useState(0);
  const [value, setValue] = React.useState(10);
  const [prediction, setPrediction] = React.useState(0);

  const [model, setModel] = React.useState<tf.Sequential | null>(null);

  const [isTraining, setIsTraining] = React.useState(false);
  const [isPredicting, setIsPredicting] = React.useState(false);

  React.useEffect(() => {
    if (model || isTraining) {
      return;
    }

    const trainModel = async () => {
      setIsTraining(true);

      const epochs = 300;

      const { model } = await loadModel({
        epochs,
        callbacks: {
          onEpochEnd: (epoch) => setTrainingProgress((epoch / epochs) * 100),
        },
      });

      setModel(model);

      setIsTraining(false);
    };

    trainModel();
  }, [isTraining, model]);

  React.useEffect(() => {
    const predict = async () => {
      if (!model) {
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

  return (
    <section className={cn("prose p-4")}>
      <section className="prose text-center">
        {isTraining ? (
          <>
            <p>Training model...</p>
            <progress
              className="progress progress-primary w-56"
              value={trainingProgress}
              max="100"
            />
          </>
        ) : (
          <>
            <input
              value={value}
              onChange={(e) => setValue(Number(e.target.value))}
              disabled={!model || isTraining}
              placeholder="Type here"
              className="input"
              type="number"
            />

            <label className="input">
              <input
                value={formatNumber(prediction)}
                type="text"
                className={cn(
                  "grow transition-all",
                  isPredicting ? "opacity-0" : "opacity-100"
                )}
                disabled
              />
              {isPredicting && <span className="loading loading-spinner" />}
            </label>
          </>
        )}
      </section>
      <MemoryUsage />
    </section>
  );
}
