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

export function splitIntoChunks(num: number, size = 2) {
  const str = String(Math.floor(num));
  const chunks = [];
  for (let i = 0; i < str.length; i += size) {
    chunks.push(Number(str.slice(i, i + size)));
  }
  return chunks;
}
