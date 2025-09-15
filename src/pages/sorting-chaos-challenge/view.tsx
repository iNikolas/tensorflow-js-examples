import React from "react";
import { TbAxisX } from "react-icons/tb";
import { TbAxisY } from "react-icons/tb";
import { SwapInput } from "@/components/ui/swap-input";

import { imageSize } from "./config";
import { useRandomnessTensor, useSortedTensor } from "./utils";

export default function Page() {
  const [swap, setSwap] = React.useState(false);
  const originalCanvasRef = React.useRef<HTMLCanvasElement>(null!);
  const sortedCanvasRef = React.useRef<HTMLCanvasElement>(null!);

  const randomnessTensor = useRandomnessTensor(originalCanvasRef.current);

  useSortedTensor({
    swap,
    randomnessTensor,
    canvas: sortedCanvasRef.current,
  });

  return (
    <section className="max-w-6xl mx-auto flex flex-col justify-center items-center gap-2 w-full p-2">
      <div className="prose">
        <h2>Sorting Chaos Challenge</h2>
        <p>
          How can you generate a random 400 x 400 grayscale tensor and then sort
          the random pixels along an axis?
        </p>

        <p>
          Click to swap which axis to sort:{" "}
          <SwapInput
            className="text-3xl"
            checked={swap}
            onChange={(e) => setSwap(e.target.checked)}
            swapOnComponent={<TbAxisY />}
            swapOffComponent={<TbAxisX />}
          />
        </p>

        <div className="flex gap-6 flex-col justify-center">
          <canvas
            width={imageSize}
            height={imageSize}
            ref={originalCanvasRef}
          />
          <canvas width={imageSize} height={imageSize} ref={sortedCanvasRef} />
        </div>
      </div>
    </section>
  );
}
