import React from "react";
import type {
  ArrayType1D,
  ArrayType2D,
} from "node_modules/danfojs/dist/danfojs-base/shared/types";

export interface ReferenceTableProps
  extends React.TableHTMLAttributes<HTMLTableElement> {
  data: { columns: string[]; data: ArrayType1D | ArrayType2D } | null;
  embarkedClasses: {
    [key: string]: number;
  };
}
