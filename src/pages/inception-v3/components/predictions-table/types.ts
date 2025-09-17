import type { PredictionResult } from "../../types";

export interface PredictionsTableProps
  extends React.TableHTMLAttributes<HTMLTableElement> {
  data: PredictionResult[];
}
