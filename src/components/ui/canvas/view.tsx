import React from "react";
import { Stage, Layer, Line, Rect } from "react-konva";

import { cn } from "@/utils/helpers";
import { useSize } from "@/utils/hooks";

import type { CanvasProps } from "./types";

export function Canvas({ className, children, lines, ...props }: CanvasProps) {
  const [container, setContainer] = React.useState<HTMLDivElement | null>(null);
  const size = useSize(container);

  return (
    <div
      ref={setContainer}
      className={cn(
        "w-full max-w-80 aspect-square relative border-4 border-primary shadow-2xl",
        className
      )}
    >
      {!!size && size.width > 0 && (
        <Stage
          width={size.width}
          height={size.height}
          className="absolute inset-0"
          {...props}
        >
          <Layer>
            <Rect
              x={0}
              y={0}
              width={size.width}
              height={size.height}
              fill="white"
            />
            {lines?.map((line, i) => (
              <Line
                key={i}
                points={line.points}
                stroke="black"
                strokeWidth={10}
                tension={0.5}
                lineCap="round"
              />
            ))}
            {children}
          </Layer>
        </Stage>
      )}
    </div>
  );
}
