import * as Yup from "yup";
import { Formik, Form, Field, ErrorMessage } from "formik";

import { cn } from "@/utils/helpers";

import { ports } from "../../config";
import type { ProfileProps } from "./types";
import { AutoSubmitOnChange, Avatar } from "./components";

export function Profile({
  className,
  onSubmit,
  limits,
  sample,
  ...props
}: ProfileProps) {
  if (!sample) {
    return null;
  }

  return (
    <Formik
      initialValues={sample}
      validationSchema={Yup.object().shape({
        age: Yup.number()
          .required("Age is Required")
          .min(limits?.min.age ?? 1, 'Input positive number for "Age" field')
          .max(
            limits?.max.age ?? 110,
            `Age must be at most ${limits?.max.age ?? 110}`
          ),
        siblingsAmount: Yup.number()
          .required("Siblings amount is Required")
          .integer('Input whole number for "Siblings amount" field')
          .min(
            limits?.min.siblingsAmount ?? 0,
            `Siblings amount must be at least ${
              limits?.min.siblingsAmount ?? 0
            }`
          )
          .max(
            limits?.max.siblingsAmount ?? 5,
            `Siblings amount must be at most ${limits?.max.siblingsAmount ?? 5}`
          ),
        familyAmount: Yup.number()
          .required("Family amount is Required")
          .integer('Input whole number for "Family amount" field')
          .min(
            limits?.min.familyAmount ?? 0,
            `Family amount must be at least ${limits?.min.familyAmount ?? 0}`
          )
          .max(
            limits?.max.familyAmount ?? 5,
            `Family amount must be at most ${limits?.max.familyAmount ?? 5}`
          ),
        fare: Yup.number()
          .required("Fare is Required")
          .min(
            limits?.min.fare ?? 0,
            `Fare must be at least ${limits?.min.fare ?? 0}`
          )
          .max(
            limits?.max.fare ?? 1000,
            `Fare must be at most ${limits?.max.fare ?? 1000}`
          ),
      })}
      onSubmit={(values, helpers) => onSubmit?.(values, helpers)}
      {...props}
    >
      <Form className={cn("hero bg-base-200", className)}>
        <AutoSubmitOnChange />
        <div className="hero-content flex-col lg:flex-row w-full">
          <Avatar />
          <fieldset className="fieldset bg-base-200 border-base-300 rounded-box border p-4 w-full">
            <legend className="fieldset-legend">
              Fill up Passenger profile
            </legend>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 [&_.label]:block [&_.label]:mb-0.5">
              <div>
                <label htmlFor="sex" className="label">
                  Sex:
                </label>
                <Field
                  type="number"
                  as="select"
                  className="select w-full"
                  id="sex"
                  name="male"
                >
                  <option value={1}>Male</option>
                  <option value={0}>Female</option>
                </Field>
              </div>

              <div>
                <label htmlFor="age" className="label">
                  Age (years):
                </label>
                <Field
                  type="number"
                  className="input w-full"
                  id="age"
                  name="age"
                />
                <ErrorMessage name="age" component="p" className="text-error" />
              </div>

              <div>
                <label htmlFor="class" className="label">
                  Passenger Class:
                </label>
                <Field
                  as="select"
                  className="select w-full"
                  id="class"
                  name="passengerClass"
                >
                  <option value={1}>First Class Passenger</option>
                  <option value={2}>Second Class Passenger</option>
                  <option value={3}>Third Class Passenger</option>
                </Field>
              </div>

              <div>
                <label htmlFor="siblings" className="label">
                  Number of siblings or spouses aboard:
                </label>
                <Field
                  type="number"
                  className="input w-full"
                  id="siblings"
                  name="siblingsAmount"
                />
                <ErrorMessage
                  name="siblingsAmount"
                  component="p"
                  className="text-error"
                />
              </div>

              <div>
                <label htmlFor="family" className="label">
                  Number of parents or children aboard:
                </label>
                <Field
                  type="number"
                  className="input w-full"
                  id="family"
                  name="familyAmount"
                />
                <ErrorMessage
                  name="familyAmount"
                  component="p"
                  className="text-error"
                />
              </div>

              <div>
                <label htmlFor="fare" className="label">
                  Passenger fare (British Pounds):
                </label>
                <Field
                  type="number"
                  className="input w-full"
                  id="fare"
                  name="fare"
                />
                <ErrorMessage
                  name="fare"
                  component="p"
                  className="text-error"
                />
                <p className="opacity-50">
                  £1 in 1912 is roughly £120–130 today
                </p>
              </div>

              <div>
                <label htmlFor="port" className="label">
                  Port of embarkation:
                </label>
                <Field
                  as="select"
                  className="select w-full"
                  id="port"
                  name="port"
                >
                  {Object.entries(ports).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </Field>
              </div>
            </div>

            <button className="btn btn-primary mt-4" type="submit">
              Get Started
            </button>
          </fieldset>
        </div>
      </Form>
    </Formik>
  );
}
