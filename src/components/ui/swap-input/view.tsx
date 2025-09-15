import { cn } from "@/utils/helpers";

import type { SwapInputProps } from "./types";

export function SwapInput({
  className,
  swapOffComponent,
  swapOnComponent,
  ...props
}: SwapInputProps) {
  return (
    <label className={cn("swap swap-flip", className)}>
      <input type="checkbox" {...props} />

      <span className="swap-on">{swapOnComponent}</span>
      <span className="swap-off">{swapOffComponent}</span>
    </label>
  );
}
