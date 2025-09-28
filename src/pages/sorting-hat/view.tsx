import React from "react";
import type Konva from "konva";
import Confetti from "react-confetti";
import { GrRevert } from "react-icons/gr";

import { cn } from "@/utils/helpers";
import { Canvas } from "@/components/ui/canvas";
import { useCanvasDrawing } from "@/utils/hooks/canvas";
import { MemoryUsage } from "@/components/containers/memory-usage";
import { QueryClientProviderWrapper } from "@/components/providers/query-client";

import { predict } from "./utils/helpers";
import { PredictionDialog } from "./components";
import type { PredictionResult } from "./types";
import { useModelQuery } from "./utils/hooks/queries";

function Page() {
  const [prediction, setPrediction] = React.useState<PredictionResult | null>(
    null
  );
  const canvasRef = React.useRef<Konva.Stage | null>(null);
  const { lines, handlers, reset, setLines } = useCanvasDrawing();

  const { isLoading, data: model, error } = useModelQuery();

  return (
    <section
      className={cn(
        "p-4 flex flex-col gap-4 items-center",
        (isLoading || !!error) && "prose text-center"
      )}
    >
      <div className="prose">
        <h2>Sorting Hat Challenge</h2>
        <p>
          You can draw some image on the canvas and the sorting hat will try to
          predict your house.
        </p>
      </div>
      {isLoading ? (
        <>
          <p>Wait before model data will load...</p>
          <p className="loading loading-bars loading-xl" />
        </>
      ) : (
        <>
          {!!error && (
            <>
              <h4>Oops, something went wrong</h4>
              <p className="text-error">{error.message}</p>
              <p>Try to refresh the page</p>
            </>
          )}
          {!error && !!model && (
            <>
              {!!prediction && <Confetti />}
              <PredictionDialog
                className="[&_.modal-box]:p-0 [&_.modal-box]:bg-transparent"
                open={!!prediction}
                onClose={() => setPrediction(null)}
                predictions={prediction}
              />
              <Canvas ref={canvasRef} {...handlers} lines={lines} />
              <div className="join">
                <button
                  disabled={!lines.length}
                  onClick={() => setLines((prev) => prev.slice(0, -1))}
                  className="btn join-item btn-neutral"
                >
                  <GrRevert />
                </button>
                <button
                  disabled={!lines.length}
                  onClick={reset}
                  className="btn join-item btn-neutral"
                >
                  Reset
                </button>
                <button
                  disabled={!lines.length}
                  className="btn btn-primary join-item"
                  onClick={() => {
                    canvasRef.current?.toImage().then((img) => {
                      if (!model) return;

                      setPrediction(predict({ model, img }));
                    });
                  }}
                >
                  Get Started
                </button>
              </div>
            </>
          )}
        </>
      )}
      <MemoryUsage className="w-full" />
    </section>
  );
}

export default function PageWrapper() {
  return (
    <QueryClientProviderWrapper>
      <Page />
    </QueryClientProviderWrapper>
  );
}
