import * as tf from "@tensorflow/tfjs";
import { modelPath } from "./constants";

tf.ready().then(() => {
  tf.tidy(() => {
    tf.loadLayersModel(modelPath).then((model) => {
      const gameStates = {
        emptyBoard: tf.zeros([9]),
        betterBlockMe: tf.tensor([-1, 0, 0, 1, 1, -1, 0, 0, -1]),
        goForTheKill: tf.tensor([1, 0, 1, 0, -1, -1, -1, 0, 1]),
      };

      const matches = tf.stack(Object.values(gameStates));

      const results = model.predict(matches);

      if (Array.isArray(results)) {
        throw new Error("Expected results to be a tensor, got an array");
      }

      results.reshape([3, 3, 3]).print();
    });
  });
});
