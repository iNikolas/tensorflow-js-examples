import React from "react";

export interface SurvivalFeedbackProps
  extends React.HTMLAttributes<HTMLElement> {
  probability: number;
  pending?: boolean;
  threshold?: number;
}
