import React from "react";
import useResizeObserver from "@react-hook/resize-observer";

export function useObjectUrl(file: File | null) {
  const [imageUrl, setImageUrl] = React.useState("");

  React.useEffect(() => {
    setImageUrl((prev) => {
      if (prev) {
        URL.revokeObjectURL(prev);
      }

      return file ? URL.createObjectURL(file) : "";
    });
  }, [file]);

  return imageUrl;
}

export function useSize(target: HTMLElement | null) {
  const [size, setSize] = React.useState<DOMRectReadOnly>();

  React.useLayoutEffect(() => {
    if (target) {
      setSize(target.getBoundingClientRect());
    }
  }, [target]);

  useResizeObserver(target, (entry) => setSize(entry.contentRect));
  return size;
}
