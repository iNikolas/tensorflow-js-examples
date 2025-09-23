import type { FormikConfig } from "formik";

import type { TrainingParams } from "@/entities/model";

export interface TrainingParametersProps
  extends Partial<FormikConfig<TrainingParams>> {
  className?: string;
}
