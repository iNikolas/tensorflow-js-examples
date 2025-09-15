import React from "react";
import * as tf from "@tensorflow/tfjs";

import { imageSize } from "../config";

export function useRandomnessTensor(canvas?: HTMLCanvasElement) {
  const [randomnessTensor, setRandomnessTensor] = React.useState<tf.Tensor2D>();

  React.useEffect(() => {
    if (randomnessTensor) {
      return;
    }

    setRandomnessTensor(
      tf.tidy(() => {
        const randomnessTensor = tf.randomUniform<tf.Rank.R2>(
          [imageSize, imageSize],
          0,
          1
        );

        return randomnessTensor;
      })
    );
  }, [randomnessTensor]);

  React.useEffect(() => {
    if (randomnessTensor) {
      tf.browser.toPixels(randomnessTensor, canvas);
    }
  }, [canvas, randomnessTensor]);

  React.useEffect(() => {
    return () => {
      if (randomnessTensor) {
        randomnessTensor.dispose();
      }
    };
  }, [randomnessTensor]);

  return randomnessTensor;
}

export function useSortedTensor({
  swap,
  randomnessTensor,
  canvas,
}: {
  swap: boolean;
  randomnessTensor?: tf.Tensor2D;
  canvas?: HTMLCanvasElement;
}) {
  React.useEffect(() => {
    if (!randomnessTensor) {
      return;
    }

    tf.tidy(() => {
      const sortedTensor = (
        swap ? randomnessTensor.transpose<tf.Tensor2D>() : randomnessTensor
      ).topk(imageSize).values;

      tf.browser.toPixels(
        swap ? sortedTensor.transpose<tf.Tensor2D>() : sortedTensor,
        canvas
      );
    });
  }, [swap, randomnessTensor, canvas]);
}
