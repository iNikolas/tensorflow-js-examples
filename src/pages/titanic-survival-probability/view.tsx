import { BiErrorAlt } from "react-icons/bi";

import { cn } from "@/utils/helpers";
import { MemoryUsage } from "@/components/containers/memory-usage";
import { TrainingProgress } from "@/components/containers/training-progress";
import { TrainingParameters } from "@/components/containers/trainng-parameters";

import "./utils";
import { useModel } from "./utils";
import { Profile } from "./components";
import { batchSize, epochs, learningRate } from "./config";

export default function Page() {
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
  } = useModel();

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
        <Profile
          onSubmit={(data) => console.log(data)}
          limits={limits}
          sample={sample}
        />
      )}
      <MemoryUsage className="w-full" />
    </section>
  );
}
