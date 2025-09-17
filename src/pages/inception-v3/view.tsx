import React from "react";

import { cn } from "@/utils/helpers";
import { useObjectUrl } from "@/utils/hooks";
import { MemoryUsage } from "@/components/containers/memory-usage";
import { QueryClientProviderWrapper } from "@/components/providers/query-client";

import { predict } from "./utils/helpers";
import type { PredictionResult } from "./types";
import { About, PredictionsTable } from "./components";
import { useModelQuery } from "./utils/hooks/queries";

function Page() {
  const [predictions, setPredictions] = React.useState<PredictionResult[]>([]);
  const [isComputing, setIsComputing] = React.useState(false);
  const [loadedImage, setLoadedImage] = React.useState<File | null>(null);

  const loadedImgSrc = useObjectUrl(loadedImage);

  const { isLoading, data, error } = useModelQuery();

  return (
    <section
      className={cn("prose p-4", (isLoading || !!error) && "text-center")}
    >
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
          {!error && !!data && (
            <>
              <h2>
                Inception V3 - neural network architecture for image
                classification
              </h2>

              <input
                type="file"
                onChange={(e) => {
                  setLoadedImage(e.target.files?.[0] ?? null);
                  setIsComputing(!!e.target.files?.[0]);
                  setPredictions([]);
                }}
                className="file-input"
              />
              {isComputing && (
                <div className="text-center">
                  <p>Wait before computation will finish...</p>
                  <p className="loading loading-spinner loading-xl"></p>
                </div>
              )}
              <PredictionsTable data={predictions} />
              {loadedImgSrc ? (
                <img
                  src={loadedImgSrc}
                  alt="Loaded image"
                  onLoad={(e) => {
                    predict({
                      model: data,
                      onSuccess: (data) => {
                        setPredictions(data);
                        setIsComputing(false);
                      },
                      img: e.currentTarget,
                    });
                  }}
                />
              ) : (
                <p>Please select an image first</p>
              )}
              <MemoryUsage />
              <About />
            </>
          )}
        </>
      )}
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
