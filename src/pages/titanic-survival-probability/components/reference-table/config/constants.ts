import {
  datasetColumns,
  ports,
} from "@/pages/titanic-survival-probability/config";

export const tableNames: Partial<
  Record<(typeof datasetColumns)[keyof typeof datasetColumns], string>
> = {
  [datasetColumns.survived]: "Survived",
  [datasetColumns.passengerClass]: "Passenger Class",
  [datasetColumns.siblingsAmount]: "Siblings Amount",
  [datasetColumns.familyAmount]: "Family Amount",
  [datasetColumns.fare]: "Fare",
  [datasetColumns.port]: "Port",
};

export const dataParsers: Partial<
  Record<
    string,
    (
      value: number,
      paramss?: {
        embarkedClasses?: {
          [key: string]: number;
        };
      }
    ) => string
  >
> = {
  [datasetColumns.survived]: (value) => (value === 0 ? "No" : "Yes"),
  Sex: (value) => (value === 0 ? "Female" : "Male"),
  [datasetColumns.port]: (value, params) => {
    const encodedPort = Object.entries(params?.embarkedClasses ?? {}).find(
      ([, index]) => index === value
    )?.[0];

    if (!encodedPort) {
      return "Not available";
    }

    return String(
      Object.fromEntries(Object.entries(ports))[encodedPort] ?? value
    );
  },
};
