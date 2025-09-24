import React from "react";
import { useFormikContext } from "formik";

import type { Profile } from "@/pages/titanic-survival-probability/types";

export function AutoSubmitOnChange() {
  const { values, submitForm } = useFormikContext<Profile>();

  const submitRef = React.useRef(submitForm);

  React.useLayoutEffect(() => {
    submitRef.current = submitForm;
  }, [submitForm]);

  React.useEffect(() => {
    submitRef.current();
  }, [values]);

  return null;
}
