import clsx from "clsx";
import type { ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(
  num: number,
  decimals = 2,
  trailingZeros = false
): string {
  const formatted = num.toFixed(decimals);

  if (!trailingZeros) {
    return Number.parseFloat(formatted).toLocaleString();
  }

  const [integerPart, decimalPart] = formatted.split(".");
  const withSeparators = Number.parseInt(integerPart).toLocaleString();

  if (!decimals) {
    return withSeparators;
  }

  return `${withSeparators}.${decimalPart}`;
}
