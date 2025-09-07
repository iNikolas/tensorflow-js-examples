import * as tf from "@tensorflow/tfjs";

tf.tidy(() => {
  const a = tf.tensor2d(
    [
      [91, 82, 13],
      [15, 23, 62],
      [25, 66, 63],
    ],
    [3, 3]
  );
  a.print();
  const b = tf.tensor2d(
    [
      [1, 23, 83],
      [33, 12, 5],
      [7, 23, 61],
    ],
    [3, 3]
  );
  b.print();
  a.matMul(b).print();
});

export default function Home() {
  return <div />;
}
