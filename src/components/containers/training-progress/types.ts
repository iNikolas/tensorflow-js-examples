export interface TrainingProgressProps
  extends React.HTMLAttributes<HTMLElement> {
  trainingProgress: number;
  loss: number;
  accuracy?: number;
}
