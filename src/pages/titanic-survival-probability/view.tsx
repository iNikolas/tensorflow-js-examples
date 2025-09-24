import React from "react";
import { BiErrorAlt } from "react-icons/bi";

import { cn } from "@/utils/helpers";
import { MemoryUsage } from "@/components/containers/memory-usage";
import { TrainingProgress } from "@/components/containers/training-progress";
import { TrainingParameters } from "@/components/containers/trainng-parameters";

import type { Profile } from "./types";
import { useModel, usePrediction } from "./utils";
import { batchSize, epochs, learningRate } from "./config";
import {
  HistoryFeedback,
  Profile as ProfileComponent,
  SurvivalFeedback,
} from "./components";

export default function Page() {
  const [value, setValue] = React.useState<Profile | null>(null);
  const {
    trainingProgress,
    loss,
    model,
    isTraining,
    train,
    error,
    accuracy,
    limits,
    sample,
    history,
  } = useModel();

  const { prediction, isPredicting } = usePrediction({ model, value });

  return (
    <section className={cn("p-4 flex flex-col gap-4 items-center")}>
      {!model &&
        (isTraining ? (
          <TrainingProgress
            trainingProgress={trainingProgress}
            loss={loss}
            accuracy={accuracy}
          />
        ) : (
          <>
            {!!error && (
              <div role="alert" className="alert alert-error">
                <BiErrorAlt size={24} />
                <span>{error}</span>
              </div>
            )}
            <TrainingParameters
              initialValues={{ epochs, learningRate, batchSize }}
              onSubmit={(values) => train(values)}
            />
          </>
        ))}
      {!!model && (
        <>
          <ProfileComponent
            onSubmit={(data) => {
              if (isPredicting) return;
              setValue(data);
            }}
            limits={limits}
            sample={sample}
          />
          <div className="flex gap-4 flex-wrap justify-center items-center">
            <HistoryFeedback className="max-sm:order-1" history={history} />
            <SurvivalFeedback
              className="self-stretch"
              pending={isPredicting}
              probability={prediction}
            />
          </div>
        </>
      )}
      <MemoryUsage className="w-full" />
    </section>
  );
}
