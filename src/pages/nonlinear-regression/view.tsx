import React from "react";

import { cn, formatNumber } from "@/utils/helpers";
import { MemoryUsage } from "@/components/containers/memory-usage";
import { TrainingProgress } from "@/components/containers/training-progress";

import { useModel, usePrediction } from "./utils";

export default function Page() {
  const [value, setValue] = React.useState(10);

  const { trainingProgress, loss, model, isTraining } = useModel();

  const { prediction, isPredicting } = usePrediction({ value, model });

  return (
    <section className={cn("prose p-4")}>
      {isTraining ? (
        <TrainingProgress trainingProgress={trainingProgress} loss={loss} />
      ) : (
        <section className="text-center">
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
        </section>
      )}
      <MemoryUsage />
    </section>
  );
}
