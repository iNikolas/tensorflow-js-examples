import { cn, splitIntoChunks } from "@/utils/helpers";

import type { CountdownProps } from "./types";

export function Countdown({
  className,
  value,
  children,
  decimals = 2,
  ...props
}: CountdownProps) {
  const fractions = Number.parseInt(
    Number.parseFloat(value.toFixed(decimals)).toString().split(".")[1]
  );

  return (
    <p className={cn("font-mono text-2xl", className)} {...props}>
      <strong>{children} </strong>
      <span className="countdown">
        {splitIntoChunks(value).map((chunk, index) => (
          <span
            key={`loss-${index}`}
            style={{ "--value": chunk } as React.CSSProperties}
            aria-live="polite"
            aria-label={chunk.toString()}
          >
            {chunk}
          </span>
        ))}
        {!!decimals && !Number.isNaN(fractions) && (
          <>
            .
            {splitIntoChunks(fractions).map((chunk, index) => (
              <span
                key={`loss-${index}`}
                style={{ "--value": chunk } as React.CSSProperties}
                aria-live="polite"
                aria-label={chunk.toString()}
              >
                {chunk}
              </span>
            ))}
          </>
        )}
      </span>
    </p>
  );
}
