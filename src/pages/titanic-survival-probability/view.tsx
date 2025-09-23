import React from "react";

import { cn } from "@/utils/helpers";
import { MemoryUsage } from "@/components/containers/memory-usage";

import "./utils";

export default function Page() {
  return (
    <section className={cn("prose p-4")}>
      <MemoryUsage />
    </section>
  );
}
