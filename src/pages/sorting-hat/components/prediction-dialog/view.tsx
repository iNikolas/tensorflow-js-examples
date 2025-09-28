import { Dialog } from "@/components/ui/dialog";

import darkMarkImg from "./assets/dark-mark.webp";
import sortingHatImg from "./assets/sorting-hat.webp";

import { houses } from "./config";
import type { PredictionDialogProps } from "./types";

export function PredictionDialog({
  predictions,
  deathEaterThreshold = 0.8,
  ...props
}: PredictionDialogProps) {
  if (!!predictions && predictions?.deatheater >= deathEaterThreshold) {
    return (
      <Dialog {...props}>
        <div className="relative flex flex-col items-center gap-6 p-6 rounded-2xl bg-gradient-to-b from-[#2a0000] to-black text-center text-white shadow-2xl">
          <img
            src={darkMarkImg}
            alt="Dark Mark"
            className="w-32 h-auto animate-pulse drop-shadow-[0_0_20px_red]"
          />
          <h2 className="text-3xl font-bold text-red-500">
            ⚠️ Dark Path Ahead!
          </h2>
          <p className="text-gray-300">
            The Sorting Hat senses a strong pull toward the Death Eaters...
          </p>
        </div>
      </Dialog>
    );
  }

  return (
    <Dialog {...props}>
      <div className="relative flex flex-col items-center justify-center gap-6 p-6 bg-gradient-to-b from-[#1a1a2e] to-[#16213e] rounded-2xl shadow-2xl">
        <img
          src={sortingHatImg}
          alt="Sorting Hat"
          className="w-2/3 h-auto drop-shadow-lg"
        />

        <h2 className="text-3xl font-bold text-white">Congratulations!</h2>
        <p className="text-gray-300">The Sorting Hat has made its choice...</p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-4">
          {houses
            .sort(
              (a, b) =>
                (predictions?.[b.name] ?? 0) - (predictions?.[a.name] ?? 0)
            )
            .map(({ name, crest, color }) => (
              <div
                key={name}
                className="flex flex-col items-center"
                style={{ opacity: predictions?.[name] ?? 0 }}
              >
                <img
                  src={crest}
                  alt={name}
                  className="w-24 h-24 drop-shadow-md"
                />
                <span
                  className="mt-2 font-semibold capitalize text-white drop-shadow-[0_0_6px_var(--tw-shadow-color)]"
                  style={
                    {
                      opacity: predictions?.[name] ?? 0,
                      "--tw-shadow-color": color,
                    } as React.CSSProperties
                  }
                >
                  {name}
                </span>
              </div>
            ))}
        </div>
      </div>
    </Dialog>
  );
}
