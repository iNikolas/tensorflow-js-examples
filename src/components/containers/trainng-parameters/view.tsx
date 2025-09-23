import * as Yup from "yup";
import { Formik, Form, Field, ErrorMessage } from "formik";

import type { TrainingParametersProps } from "./types";

export function TrainingParameters({
  className,
  onSubmit,
  ...props
}: TrainingParametersProps) {
  return (
    <Formik
      initialValues={{ batchSize: 32, epochs: 100, learningRate: 0.01 }}
      onSubmit={(values, helpers) => onSubmit?.(values, helpers)}
      validationSchema={Yup.object({
        batchSize: Yup.number()
          .required("Batch size is Required")
          .min(1, "Batch size must be at least 1")
          .max(512, "Batch size must be at most 512"),
        epochs: Yup.number()
          .required("Epochs is Required")
          .min(1, "Epochs must be at least 1")
          .max(1000, "Epochs must be at most 1000"),
        learningRate: Yup.number()
          .required()
          .min(0, "Learning rate must be at least 0")
          .max(1, "Learning rate must be at most 1"),
      })}
      {...props}
    >
      <Form className={className}>
        <fieldset className="fieldset bg-base-200 border-base-300 rounded-box w-xs border p-4">
          <legend className="fieldset-legend">
            Adjust training parameters
          </legend>

          <label htmlFor="batch" className="label">
            Batch size
          </label>
          <Field type="number" className="input" id="batch" name="batchSize" />
          <ErrorMessage name="batchSize" component="p" className="text-error" />
          <p className="text-xs opacity-70 mt-1">
            Small batches (8–32) = stable but slower. Large batches (&gt;64) =
            faster but may hurt generalization.
          </p>

          <label htmlFor="epochs" className="label mt-3">
            Epochs
          </label>
          <Field type="number" className="input" id="epochs" name="epochs" />
          <ErrorMessage name="epochs" component="p" className="text-error" />
          <p className="text-xs opacity-70 mt-1">
            One epoch = one full dataset pass. Too few → underfit. Too many →
            overfit.
          </p>

          <label htmlFor="learning-rate" className="label mt-3">
            Learning Rate
          </label>
          <Field
            type="number"
            className="input"
            placeholder="e.g. 0.01 or 0.001"
            id="learning-rate"
            name="learningRate"
          />
          <ErrorMessage
            name="learningRate"
            component="p"
            className="text-error"
          />
          <p className="text-xs opacity-70 mt-1">
            Higher rate = faster start but risk overshooting. Lower = safer but
            slower.
          </p>

          <button className="btn btn-primary mt-4" type="submit">
            Train
          </button>
        </fieldset>
      </Form>
    </Formik>
  );
}
