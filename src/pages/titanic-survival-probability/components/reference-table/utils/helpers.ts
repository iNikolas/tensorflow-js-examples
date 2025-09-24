import { tableNames } from "../config";

export function getColumnName(column: string) {
  return Object.fromEntries(Object.entries(tableNames))[column] ?? column;
}
