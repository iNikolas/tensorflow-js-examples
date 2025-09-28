import type Konva from "konva";
import type { StageProps } from "react-konva";

type LineData = { points: number[] };

export interface CanvasProps extends StageProps {
  lines?: LineData[];
  ref?: React.RefObject<Konva.Stage | null>;
}
