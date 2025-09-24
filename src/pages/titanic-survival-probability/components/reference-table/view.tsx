import { cn } from "@/utils/helpers";

import { getColumnName } from "./utils";
import type { ReferenceTableProps } from "./types";
import { dataParsers } from "./config";

export function ReferenceTable({
  className,
  data,
  embarkedClasses,
  ...props
}: ReferenceTableProps) {
  if (!data) {
    return null;
  }

  return (
    <div className={cn("overflow-auto max-h-96 max-w-[74vw]", className)}>
      <table
        className="table table-xs table-pin-rows table-pin-cols"
        {...props}
      >
        <thead>
          <tr>
            {data.columns.map((column) => (
              <td key={column}>{getColumnName(column)}</td>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.data.map(
            (row, index) =>
              Array.isArray(row) && (
                <tr key={`row-${index}`}>
                  {row.map((cell, index) => (
                    <td key={`cell-${index}`}>
                      {dataParsers[data.columns[index]]?.(Number(cell), {
                        embarkedClasses,
                      }) ?? cell}
                    </td>
                  ))}
                </tr>
              )
          )}
        </tbody>
        <tfoot>
          <tr>
            {data.columns.map((column) => (
              <td key={column}>{getColumnName(column)}</td>
            ))}
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
