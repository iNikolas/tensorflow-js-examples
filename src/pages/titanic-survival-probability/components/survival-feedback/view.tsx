import { cn, formatNumber } from "@/utils/helpers";
import { BiCheckCircle, BiXCircle } from "react-icons/bi";

import type { SurvivalFeedbackProps } from "./types";

export function SurvivalFeedback({
  probability,
  pending,
  className,
  threshold = 0.5,
  ...props
}: SurvivalFeedbackProps) {
  const isPositive = probability >= threshold;
  return (
    <section
      className={cn(
        "stat max-w-max",
        !pending &&
          (isPositive
            ? "bg-gradient-to-r from-green-100/50 to-green-200/50 border-l-4 border-green-500"
            : "bg-gradient-to-r from-red-100/50 to-red-200/50 border-l-4 border-red-500"),
        className
      )}
      {...props}
    >
      <div
        className={cn(
          "stat-figure",
          !pending && (isPositive ? "text-success" : "text-error")
        )}
      >
        {!pending &&
          (isPositive ? <BiCheckCircle size={28} /> : <BiXCircle size={28} />)}
        {pending && <div className="loading loading-spinner w-7 h-7" />}
      </div>
      <div className="stat-title">Probability</div>
      <p
        className={cn(
          "stat-value",
          !pending && (isPositive ? "text-success" : "text-error")
        )}
      >
        <span className={cn(pending && "skeleton [&>*]:opacity-0")}>
          <span className="transition-all opacity-100">
            {formatNumber(probability * 100)}
          </span>
        </span>{" "}
        %
      </p>
      <p className={cn("stat-desc", pending && "skeleton [&>*]:opacity-0")}>
        <span className="transition-all opacity-100">
          {isPositive ? "Chance of survival" : "Chance of not surviving"}
        </span>
      </p>
    </section>
  );
}
