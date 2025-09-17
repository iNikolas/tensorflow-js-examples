import { cn } from "@/utils/helpers";

import { thresholdConfidence } from "../../config";
import type { PredictionsTableProps } from "./types";

export function PredictionsTable({
  className,
  data,
  ...props
}: PredictionsTableProps) {
  if (!data.length) {
    return null;
  }

  return (
    <div
      className={cn(
        "overflow-x-auto rounded-box border border-base-content/5 bg-base-100 p-1 max-h-60",
        className
      )}
    >
      <table className="table w-full table-pin-rows" {...props}>
        <thead>
          <tr>
            <th>Label</th>
            <th>Probability</th>
          </tr>
        </thead>
        <tbody>
          {data.map(({ label, confidence }) => (
            <tr key={label}>
              <td className="capitalize">{label}</td>
              <td
                className={cn(
                  confidence <= thresholdConfidence && "text-error"
                )}
              >
                {confidence}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
