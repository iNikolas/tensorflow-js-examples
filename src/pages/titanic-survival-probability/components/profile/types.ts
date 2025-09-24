import type { FormikConfig } from "formik";

import type { Limits, Profile } from "../../types";

export interface ProfileProps extends Partial<FormikConfig<Profile>> {
  className?: string;
  limits: Limits | null;
  sample: Profile | null;
}
