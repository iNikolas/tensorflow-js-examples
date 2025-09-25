import React from "react";
import * as tf from "@tensorflow/tfjs";
import type { MinMaxScaler } from "danfojs";

import type { TrainingParams } from "@/entities/model";

import {
  calculateLimits,
  loadModel,
  profileToSample,
  sampleToProfile,
} from "./helpers";
import { uiUpdateIntervalMs } from "../config";
import type { Limits, Profile } from "../types";
import type {
  ArrayType1D,
  ArrayType2D,
} from "node_modules/danfojs/dist/danfojs-base/shared/types";

export function useModel() {
  const [scaler, setScaler] = React.useState<MinMaxScaler | null>(null);
  const [error, setError] = React.useState("");
  const [trainingProgress, setTrainingProgress] = React.useState(0);
  const [loss, setLoss] = React.useState(Infinity);
  const [accuracy, setAccuracy] = React.useState(Infinity);
  const [sample, setSample] = React.useState<Profile | null>(null);
  const [history, setHistory] = React.useState<tf.History | null>(null);
  const [columns, setColumns] = React.useState<string[]>([]);
  const [embarkedClasses, setEmbarkedClasses] = React.useState<{
    [key: string]: number;
  }>({});
  const [referenceData, setReferenceData] = React.useState<{
    columns: string[];
    data: ArrayType1D | ArrayType2D;
  } | null>(null);

  const updatedTimestampRef = React.useRef(0);

  const [model, setModel] = React.useState<tf.Sequential | null>(null);
  const [limits, setLimits] = React.useState<Limits | null>(null);

  const [isTraining, setIsTraining] = React.useState(false);

  React.useEffect(() => {
    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, [isTraining, model]);

  return {
    trainingProgress,
    loss,
    model,
    isTraining,
    error,
    accuracy,
    limits,
    sample,
    history,
    columns,
    embarkedClasses,
    scaler,
    referenceData,
    train: async (params: TrainingParams) => {
      if (model || isTraining) {
        return;
      }

      setIsTraining(true);

      try {
        const {
          model: data,
          scaler,
          columns,
          sample,
          embarkedClasses,
          history,
          referenceData,
        } = await loadModel({
          callbacks: {
            onEpochEnd: (epoch, log) => {
              setTrainingProgress((epoch / params.epochs) * 100);

              const currentTimestamp = Date.now();

              if (
                currentTimestamp - updatedTimestampRef.current >
                uiUpdateIntervalMs
              ) {
                updatedTimestampRef.current = currentTimestamp;

                setLoss(log?.loss ?? Infinity);
                setAccuracy(log?.val_acc ?? 0);
              }
            },
          },
          ...params,
        });

        setReferenceData(referenceData);
        setScaler(scaler);
        setEmbarkedClasses(embarkedClasses);
        setColumns(columns);
        setHistory(history);
        setSample(sampleToProfile({ sample, columns, embarkedClasses }));
        setLimits(calculateLimits({ scaler, columns }));
        setModel(data);
      } catch (error) {
        setError(
          error instanceof Error ? error.message : JSON.stringify(error)
        );
      } finally {
        setIsTraining(false);
      }
    },
  };
}

export function usePrediction({
  value,
  model,
  columns,
  embarkedClasses,
  scaler,
}: {
  value: Profile | null;
  model?: tf.Sequential | null;
  columns: string[];
  embarkedClasses: { [key: string]: number };
  scaler: MinMaxScaler | null;
}) {
  const [prediction, setPrediction] = React.useState(0);

  const [isPredicting, setIsPredicting] = React.useState(false);

  const columnsRef = React.useRef(columns);

  const embarkedClassesRef = React.useRef(embarkedClasses);

  const scalerRef = React.useRef(scaler);

  React.useLayoutEffect(() => {
    scalerRef.current = scaler;
  }, [scaler]);

  React.useLayoutEffect(() => {
    columnsRef.current = columns;
  }, [columns]);

  React.useLayoutEffect(() => {
    embarkedClassesRef.current = embarkedClasses;
  }, [embarkedClasses]);

  React.useEffect(() => {
    const predict = async () => {
      if (!model || value == null) {
        return;
      }

      try {
        setIsPredicting(true);

        const data = profileToSample({
          profile: value,
          columns: columnsRef.current,
          embarkedClasses: embarkedClassesRef.current,
        });

        const prediction = tf.tidy(() =>
          model.predict(tf.tensor1d(data.slice(1)).expandDims())
        );

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
