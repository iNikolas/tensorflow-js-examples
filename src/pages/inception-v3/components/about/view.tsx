import React from "react";

import { cn } from "@/utils/helpers";

export function About({
  className,
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  return (
    <section
      className={cn(
        "collapse bg-base-100 border-base-300 border collapse-arrow",
        className
      )}
      {...props}
    >
      <input type="checkbox" />
      <div className="collapse-title font-semibold">
        How to achieve precise predictions
      </div>
      <div className="collapse-content">
        <h6>1️⃣ Number of objects per image</h6>
        <ul>
          <li>Single object per image → Best results.</li>
          <ul>
            <li>
              Inception V3 is optimized for recognizing one primary object.
            </li>
          </ul>
          <li>Multiple objects → Accuracy drops.</li>
          <ul>
            <li>
              The model may predict the most prominent object, or mix
              predictions from different objects.
            </li>
            <li>
              If you want multiple object recognition, consider object detection
              models (e.g., Faster R-CNN, YOLO, SSD) instead.
            </li>
          </ul>
        </ul>

        <h6>2️⃣ Object size and placement</h6>
        <ul>
          <li>Centered, large object → More precise predictions.</li>
          <li>
            Small or off-center object → Model might miss it or focus on
            background.
          </li>
          <li>Cropping the object before feeding it improves precision.</li>
        </ul>

        <h6>3️⃣ Background</h6>
        <ul>
          <li>Simple, uncluttered backgrounds → Better predictions.</li>
          <li>
            Busy or similar-colored background → Model can confuse background
            with object features.
          </li>
          <li>
            Pretrained Inception V3 isn’t trained to ignore backgrounds, so
            segmentation or cropping helps.
          </li>
        </ul>

        <h6>4️⃣ Image quality</h6>
        <ul>
          <li>High resolution, sharp images → Higher accuracy.</li>
          <li>
            Blur, noise, compression artifacts → Reduces confidence and may
            misclassify.
          </li>
          <li>
            Consistent lighting helps because Inception V3 isn’t robust to
            extreme lighting changes.
          </li>
        </ul>

        <h6>5️⃣ Object viewpoint and orientation</h6>
        <ul>
          <li>
            Canonical views (like ImageNet standard: front, side, or typical
            pose) → Better.
          </li>
          <li>
            Extreme rotations, unusual angles, occlusions → Accuracy drops.
          </li>
        </ul>

        <h6>⚡ Summary (Photo-wise)</h6>
        <table className="table">
          <tbody>
            <tr>
              <th>Factor</th>
              <th>Ideal for Inception V3</th>
            </tr>
            <tr>
              <td>Objects per image</td>
              <td>Single</td>
            </tr>
            <tr>
              <td>Object placement</td>
              <td>Centered</td>
            </tr>
            <tr>
              <td>Object size</td>
              <td>Large relative to frame</td>
            </tr>
            <tr>
              <td>Background</td>
              <td>Simple / clean</td>
            </tr>
            <tr>
              <td>Image quality</td>
              <td>Sharp, good lighting</td>
            </tr>
            <tr>
              <td>Object viewpoint</td>
              <td>Standard angles</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}
