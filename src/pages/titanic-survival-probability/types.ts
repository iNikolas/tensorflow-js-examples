export interface Profile {
  male: 0 | 1;
  age: number;
  passengerClass: number;
  siblingsAmount: number;
  familyAmount: number;
  fare: number;
  port: string;
}

export interface InferenceParams {
  Pclass: number;
  SibSp: number;
  Parch: number;
  Fare: number;
  Embarked: number;
  male: 0 | 1;
  female: 0 | 1;
  Age_bucket: number;
}

export interface Limits {
  min: Partial<Record<keyof Profile, number>>;
  max: Partial<Record<keyof Profile, number>>;
}
