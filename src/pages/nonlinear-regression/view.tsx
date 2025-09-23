import React from "react";

import { cn, formatNumber } from "@/utils/helpers";
import { MemoryUsage } from "@/components/containers/memory-usage";

import { splitIntoChunks, useModel, usePrediction } from "./utils";

export default function Page() {
  const [value, setValue] = React.useState(10);

  const { trainingProgress, loss, model, isTraining } = useModel();

  const { prediction, isPredicting } = usePrediction({ value, model });

  return (
    <section className={cn("prose p-4")}>
      <section className="prose text-center">
        {isTraining ? (
          <>
            <p>Training model... ({formatNumber(trainingProgress)} %)</p>
            <progress
              className="progress progress-primary w-56"
              value={trainingProgress}
              max="100"
            />
            {loss !== Infinity && (
              <p className="font-mono text-2xl">
                <strong>Loss: </strong>
                <span className="countdown">
                  {splitIntoChunks(loss).map((chunk, index) => (
                    <span
                      key={`loss-${index}`}
                      style={{ "--value": chunk } as React.CSSProperties}
                      aria-live="polite"
                      aria-label={chunk.toString()}
                    >
                      {chunk}
                    </span>
                  ))}
                  .
                  {splitIntoChunks((loss - Math.floor(loss)) * 100).map(
                    (chunk, index) => (
                      <span
                        key={`loss-${index}`}
                        style={{ "--value": chunk } as React.CSSProperties}
                        aria-live="polite"
                        aria-label={chunk.toString()}
                      >
                        {chunk}
                      </span>
                    )
                  )}
                </span>
              </p>
            )}
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
