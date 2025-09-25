import React from "react";
import { Helmet } from "react-helmet";
import { BiErrorAlt } from "react-icons/bi";
import { useDebouncedCallback } from "use-debounce";

import { cn } from "@/utils/helpers";
import { MemoryUsage } from "@/components/containers/memory-usage";
import { HistoryFeedback } from "@/components/containers/history-feedback";
import { TrainingProgress } from "@/components/containers/training-progress";
import { TrainingParameters } from "@/components/containers/trainng-parameters";

import {
  Profile as ProfileComponent,
  ReferenceTable,
  SurvivalFeedback,
} from "./components";
import type { Profile } from "./types";
import iconSrc from "./assets/titanic.ico";
import { useModel, usePrediction } from "./utils";
import { batchSize, epochs, learningRate } from "./config";

export default function Page() {
  const [value, setValue] = React.useState<Profile | null>(null);

  const debounced = useDebouncedCallback((data: Profile) => {
    setValue({ ...data });
  }, 100);

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
    columns,
    embarkedClasses,
    scaler,
    referenceData,
  } = useModel();

  const { prediction, isPredicting } = usePrediction({
    model,
    value,
    columns,
    embarkedClasses,
    scaler,
  });

  return (
    <>
      <Helmet>
        <title>Titanic Survival Probability</title>
        <meta
          name="description"
          content="Predict survival probability of Titanic passengers"
        />
        <link rel="icon" href={iconSrc} />
      </Helmet>
      <section className="prose p-4 text-center">
        <h2>Titanic Survival Probability</h2>
        <p>Predict survival probability of Titanic passengers</p>
      </section>
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
              onSubmit={debounced}
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
            <div className="prose">
              <h6>Reference data from real Titanic passengers:</h6>
              <ReferenceTable
                data={referenceData}
                embarkedClasses={embarkedClasses}
              />
            </div>
          </>
        )}
        <MemoryUsage className="w-full" />
      </section>
    </>
  );
}
