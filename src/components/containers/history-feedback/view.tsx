import { cn, formatNumber } from "@/utils/helpers";
import { BiCheckCircle, BiXCircle } from "react-icons/bi";
import type { HistoryFeedbackProps } from "./types";

export function HistoryFeedback({
  className,
  history,
  threshold = 0.8,
  ...props
}: HistoryFeedbackProps) {
  if (!history) {
    return null;
  }

  const accHistory = history.history?.acc;
  const valAccHistory = history.history?.val_acc;
  const lossHistory = history.history?.loss;
  const valLossHistory = history.history?.val_loss;

  const finalAccRaw = accHistory?.at(-1) ?? 0;
  const finalValAcc = valAccHistory?.at(-1) ?? null;
  const finalLoss = lossHistory?.at(-1) ?? null;
  const finalValLoss = valLossHistory?.at(-1) ?? null;

  const finalAcc = typeof finalAccRaw === "number" ? finalAccRaw : 0;

  const isGood = finalAcc >= threshold;

  return (
    <section
      className={cn(
        "stat max-w-max transition-all",
        isGood
          ? "bg-gradient-to-r from-green-100/50 to-green-200/50 border-l-4 border-green-500"
          : "bg-gradient-to-r from-red-100/50 to-red-200/50 border-l-4 border-red-500",
        className
      )}
      {...props}
    >
      <div
        className={cn("stat-figure", isGood ? "text-success" : "text-error")}
      >
        {isGood ? <BiCheckCircle size={28} /> : <BiXCircle size={28} />}
      </div>

      <div className="stat-title">Training Results</div>

      <p
        className={cn("stat-value", isGood ? "text-green-700" : "text-red-700")}
      >
        {formatNumber(finalAcc * 100, 0)}% acc
      </p>

      {finalValAcc !== null && typeof finalValAcc === "number" && (
        <p className="stat-desc">
          Validation: {formatNumber(finalValAcc * 100, 0)}% acc
        </p>
      )}

      {finalLoss !== null && typeof finalLoss === "number" && (
        <p className="stat-desc">
          Loss: {formatNumber(finalLoss, 3)}
          {finalValLoss !== null &&
            typeof finalValLoss === "number" &&
            ` / Val Loss: ${formatNumber(finalValLoss, 3)}`}
        </p>
      )}
    </section>
  );
}
