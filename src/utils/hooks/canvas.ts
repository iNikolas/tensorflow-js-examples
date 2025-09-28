import React from "react";
import type Konva from "konva";

export function useCanvasDrawing() {
  const [lines, setLines] = React.useState<{ points: number[] }[]>([]);
  const [isDrawing, setIsDrawing] = React.useState(false);

  const handleMouseDown = (
    e: Konva.KonvaEventObject<MouseEvent | TouchEvent>
  ) => {
    setIsDrawing(true);
    const pos = e.target.getStage()?.getPointerPosition();
    if (pos) setLines([...lines, { points: [pos.x, pos.y] }]);
  };

  const handleMouseMove = (
    e: Konva.KonvaEventObject<MouseEvent | TouchEvent>
  ) => {
    if (!isDrawing) return;
    const stage = e.target.getStage();
    const point = stage?.getPointerPosition();
    if (!point) return;
    const lastLine = lines[lines.length - 1];
    lastLine.points = lastLine.points.concat([point.x, point.y]);
    setLines([...lines.slice(0, -1), lastLine]);
  };

  const handleMouseUp = () => setIsDrawing(false);

  return {
    lines,
    handlers: {
      onMouseDown: handleMouseDown,
      onMousemove: handleMouseMove,
      onMouseup: handleMouseUp,
      onTouchStart: handleMouseDown,
      onTouchMove: handleMouseMove,
      onTouchEnd: handleMouseUp,
    },
    reset: () => setLines([]),
    setLines,
  };
}
