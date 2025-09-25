export const validationSplit = 0;

export const learningRate = 0.1;

export const batchSize = 32;

export const epochs = 100;

export const uiUpdateIntervalMs = 2000;

export const ageGroups = [
  { max: 10, label: "Children" },
  { max: 40, label: "Young adults" },
  { max: Infinity, label: "Elderly" },
] as const;

export const ports = {
  C: "Cherbourg",
  Q: "Queenstown",
  S: "Southampton",
} as const;

export const datasetColumns = {
  survived: "Survived",
  passengerClass: "Pclass",
  siblingsAmount: "SibSp",
  familyAmount: "Parch",
  fare: "Fare",
  port: "Embarked",
  male: "male",
  female: "female",
  ageBucket: "Age_bucket",
} as const;
