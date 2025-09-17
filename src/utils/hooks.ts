import React from "react";

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
