import React from "react";

export interface SwapInputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  swapOnComponent: React.ReactNode;
  swapOffComponent: React.ReactNode;
}
