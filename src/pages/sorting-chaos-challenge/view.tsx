import React from "react";
import * as tf from "@tensorflow/tfjs";

const imageSize = 400;

export default function Page() {
  const originalCanvasRef = React.useRef<HTMLCanvasElement>(null!);
  const sortedCanvasRef = React.useRef<HTMLCanvasElement>(null!);

  React.useEffect(() => {
    if (!originalCanvasRef.current) {
      return;
    }

    tf.tidy(() => {
      const randomnessTensor = tf.randomUniform<tf.Rank.R2>(
        [imageSize, imageSize],
        0,
        1
      );

      tf.browser.toPixels(randomnessTensor, originalCanvasRef.current);

      tf.browser.toPixels(
        randomnessTensor.topk(imageSize).values,
        sortedCanvasRef.current
      );
    });
  }, []);

  return (
    <section className="max-w-6xl mx-auto flex flex-col justify-center items-center gap-2 w-full p-2">
      <div className="prose">
        <h2>Sorting Chaos Challenge</h2>
        <p>
          How can you generate a random 400 x 400 grayscale tensor and then sort
          the random pixels along an axis?
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
