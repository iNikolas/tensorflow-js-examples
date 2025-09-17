import React from "react";
import * as tf from "@tensorflow/tfjs";

import { useObjectUrl } from "@/utils/hooks";

export default function Page() {
  const [loadedImage, setLoadedImage] = React.useState<File | null>(null);
  const [croppedImage0, setCroppedImage0] = React.useState<File | null>(null);
  const [croppedImage1, setCroppedImage1] = React.useState<File | null>(null);
  const [croppedImage2, setCroppedImage2] = React.useState<File | null>(null);
  const [croppedImage3, setCroppedImage3] = React.useState<File | null>(null);
  const [flippedImage, setFlippedImage] = React.useState<File | null>(null);

  const [nnResizedImage, setNNResizedImage] = React.useState<File | null>(null);
  const [bilinearResizedImage, setBilinearResizedImage] =
    React.useState<File | null>(null);

  const loadedImageUrl = useObjectUrl(loadedImage);
  const croppedImageUrl0 = useObjectUrl(croppedImage0);
  const croppedImageUrl1 = useObjectUrl(croppedImage1);
  const croppedImageUrl2 = useObjectUrl(croppedImage2);
  const croppedImageUrl3 = useObjectUrl(croppedImage3);
  const flippedImageUrl = useObjectUrl(flippedImage);
  const nnResizedImageUrl = useObjectUrl(nnResizedImage);
  const bilinearResizedImageUrl = useObjectUrl(bilinearResizedImage);

  React.useEffect(() => {
    if (!loadedImageUrl) {
      return;
    }

    const img = document.createElement("img");
    img.src = loadedImageUrl;

    img.onload = () => {
      tf.tidy(() => {
        const imageTensor: tf.Tensor4D = tf.browser
          .fromPixels(img)
          .asType("float32")
          .expandDims();

        const [red, green, blue] = tf.split(imageTensor.squeeze(), 3, 2);
        const gray = red.mul(0.299).add(green.mul(0.587)).add(blue.mul(0.114));

        const croppedImageTensor3 = tf.stack([gray, gray, gray], 2);

        const croppedSize = [img.height, img.width, 1];

        const croppedImageTensor0 = imageTensor
          .squeeze()
          .slice([0, 0, 0], croppedSize);

        const croppedImageTensor1 = imageTensor
          .squeeze()
          .slice([0, 0, 1], croppedSize);

        const croppedImageTensor2 = imageTensor
          .squeeze()
          .slice([0, 0, 2], croppedSize);

        const nnResizeTensor = tf.image.resizeNearestNeighbor(
          imageTensor,
          [img.height * 4, img.width * 4],
          true
        );

        const bilinearResizeTensor = tf.image.resizeBilinear(
          imageTensor,
          [img.height * 4, img.width * 4],
          true
        );

        const croppedCanvas = document.createElement("canvas");
        const canvas = document.createElement("canvas");
        const scaledCanvasNn = document.createElement("canvas");
        const scaledCanvasBilinear = document.createElement("canvas");

        croppedCanvas.width = croppedSize[1];
        croppedCanvas.height = croppedSize[0];

        scaledCanvasNn.width = img.width * 4;
        scaledCanvasNn.height = img.height * 4;

        scaledCanvasBilinear.width = img.width * 4;
        scaledCanvasBilinear.height = img.height * 4;

        canvas.width = img.width;
        canvas.height = img.height;

        tf.browser
          .toPixels(
            croppedImageTensor3.squeeze<tf.Tensor3D>().asType("int32"),
            croppedCanvas
          )
          .then(() => {
            croppedCanvas.toBlob((blob) => {
              if (!blob) {
                return;
              }

              setCroppedImage3(
                new File([blob], "loaded.png", {
                  type: "image/png",
                })
              );
            });
          });

        tf.browser
          .toPixels(
            croppedImageTensor0.squeeze<tf.Tensor3D>().asType("int32"),
            croppedCanvas
          )
          .then(() => {
            croppedCanvas.toBlob((blob) => {
              if (!blob) {
                return;
              }

              setCroppedImage0(
                new File([blob], "loaded.png", {
                  type: "image/png",
                })
              );
            });
          });

        tf.browser
          .toPixels(
            croppedImageTensor1.squeeze<tf.Tensor3D>().asType("int32"),
            croppedCanvas
          )
          .then(() => {
            croppedCanvas.toBlob((blob) => {
              if (!blob) {
                return;
              }

              setCroppedImage1(
                new File([blob], "loaded.png", {
                  type: "image/png",
                })
              );
            });
          });

        tf.browser
          .toPixels(
            croppedImageTensor2.squeeze<tf.Tensor3D>().asType("int32"),
            croppedCanvas
          )
          .then(() => {
            croppedCanvas.toBlob((blob) => {
              if (!blob) {
                return;
              }

              setCroppedImage2(
                new File([blob], "loaded.png", {
                  type: "image/png",
                })
              );
            });
          });

        tf.browser
          .toPixels(
            tf.image
              .flipLeftRight(imageTensor)
              .squeeze<tf.Tensor3D>()
              .asType("int32"),
            canvas
          )
          .then(() => {
            canvas.toBlob((blob) => {
              if (!blob) {
                return;
              }

              setFlippedImage(
                new File([blob], "flipped.png", {
                  type: "image/png",
                })
              );
            });
          });

        tf.browser
          .toPixels(
            nnResizeTensor.squeeze<tf.Tensor3D>().asType("int32"),
            scaledCanvasNn
          )
          .then(() => {
            scaledCanvasNn.toBlob((blob) => {
              if (!blob) {
                return;
              }

              setNNResizedImage(
                new File([blob], "nn-resized.png", {
                  type: "image/png",
                })
              );
            });
          });

        tf.browser
          .toPixels(
            bilinearResizeTensor.squeeze<tf.Tensor3D>().asType("int32"),
            scaledCanvasBilinear
          )
          .then(() => {
            scaledCanvasBilinear.toBlob((blob) => {
              if (!blob) {
                return;
              }

              setBilinearResizedImage(
                new File([blob], "bilinear-resized.png", {
                  type: "image/png",
                })
              );
            });
          });
      });
    };
  }, [loadedImageUrl]);

  return (
    <section className="max-w-6xl mx-auto flex flex-col justify-center items-center gap-2 w-full p-2">
      <div className="prose">
        <h2>Manipulating with images</h2>
      </div>
      <input
        type="file"
        onChange={(e) => setLoadedImage(e.target.files?.[0] ?? null)}
        className="file-input min-h-max"
      />
      {!!loadedImageUrl && (
        <div className="max-h-[75vh] grid grid-cols-12 grid-rows-12 gap-4 [&>*]:col-span-6 [&>*]:row-span-12 [&>*]:flex [&>*]:items-center [&_img]:object-contain [&_img]:max-h-full [&_img]:max-w-full">
          <div className="justify-end">
            <img src={loadedImageUrl} />
          </div>
          <div>{!!flippedImageUrl && <img src={flippedImageUrl} />}</div>
        </div>
      )}
      {!!nnResizedImageUrl && (
        <div className="max-h-[75vh] grid grid-cols-12 grid-rows-12 gap-4 [&>*]:col-span-6 [&>*]:row-span-12 [&>*]:flex [&>*]:items-center [&_img]:object-contain [&_img]:max-h-full [&_img]:max-w-full">
          <div className="justify-end">
            <img src={nnResizedImageUrl} />
          </div>
          <div>
            {!!bilinearResizedImageUrl && <img src={bilinearResizedImageUrl} />}
          </div>
        </div>
      )}
      {!!croppedImageUrl0 && (
        <div className="max-h-[75vh] grid grid-cols-12 grid-rows-12 gap-4 [&>*]:col-span-6 [&>*]:row-span-12 [&>*]:flex [&>*]:items-center [&_img]:object-contain [&_img]:max-h-full [&_img]:max-w-full">
          <div className="justify-end">
            <img src={croppedImageUrl0} />
          </div>
          <div>{!!croppedImageUrl1 && <img src={croppedImageUrl1} />}</div>
        </div>
      )}
      {!!croppedImageUrl2 && (
        <div className="max-h-[75vh] grid grid-cols-12 grid-rows-12 gap-4 [&>*]:col-span-6 [&>*]:row-span-12 [&>*]:flex [&>*]:items-center [&_img]:object-contain [&_img]:max-h-full [&_img]:max-w-full">
          <div className="justify-end">
            <img src={croppedImageUrl2} />
          </div>
          <div>{!!croppedImageUrl3 && <img src={croppedImageUrl3} />}</div>
        </div>
      )}
    </section>
  );
}
