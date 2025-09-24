import type { Profile } from "../../../types";

import nobleManImg from "../assets/noble-man.avif";
import maleWorkerImg from "../assets/male-worker.webp";
import frankGordonImg from "../assets/frank-gordon.webp";
import prettyWomanImg from "../assets/pretty-woman.webp";
import femaleWorkerImg from "../assets/female-worker.webp";
import edwardianMaidsImg from "../assets/edwardian-maids.webp";

export const imagesMap: Record<
  Profile["male"],
  Record<Profile["passengerClass"], string>
> = {
  0: {
    1: prettyWomanImg,
    2: femaleWorkerImg,
    3: edwardianMaidsImg,
  },
  1: {
    1: nobleManImg,
    2: frankGordonImg,
    3: maleWorkerImg,
  },
};
