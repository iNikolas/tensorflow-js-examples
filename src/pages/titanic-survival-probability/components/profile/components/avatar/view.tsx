import React from "react";
import { useFormikContext } from "formik";

import { cn } from "@/utils/helpers";
import type { Profile } from "@/pages/titanic-survival-probability/types";

import { imagesMap } from "./config";

export function Avatar({
  className,
  ...props
}: React.ImgHTMLAttributes<HTMLImageElement>) {
  const { values } = useFormikContext<Profile>();

  return (
    <img
      src={imagesMap[values.male][values.passengerClass]}
      alt="Passenger avatar"
      className={cn(
        "max-w-xs w-full aspect-[0.8] object-cover rounded-lg shadow-2xl",
        className
      )}
      {...props}
    />
  );
}
