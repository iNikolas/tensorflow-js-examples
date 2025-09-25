import { cn, formatNumber } from "@/utils/helpers";

import { Countdown } from "./components";
import type { TrainingProgressProps } from "./types";

export function TrainingProgress({
  loss,
  accuracy,
  trainingProgress,
  className,
  ...props
}: TrainingProgressProps) {
  return (
    <section className={cn("prose text-center", className)} {...props}>
      <p>Training model... ({formatNumber(trainingProgress)} %)</p>
      <progress
        className="progress progress-primary w-56"
        value={trainingProgress}
        max="100"
      />
      {loss !== Infinity && (
        <div>
          <Countdown className="mb-0" decimals={4} value={loss}>
            Loss:
          </Countdown>
          <p className="text-sm opacity-50 mt-0">
            Training will stop earlier if loss does not decrease
          </p>
        </div>
      )}
      {accuracy != null && accuracy !== Infinity && (
        <Countdown decimals={1} value={accuracy * 100}>
          Accuracy(%):
        </Countdown>
      )}
    </section>
  );
}
