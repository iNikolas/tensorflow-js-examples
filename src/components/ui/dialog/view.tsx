import React from "react";

import { cn } from "@/utils/helpers";

import { useDialog } from "./utils";

export function Dialog({
  className,
  open,
  children,
  ...props
}: React.DialogHTMLAttributes<HTMLDialogElement>) {
  const dialogRef = useDialog(open);

  return (
    <dialog ref={dialogRef} className={cn("modal", className)} {...props}>
      <div className="modal-box">{children}</div>
      <form method="dialog" className="modal-backdrop">
        <button>close</button>
      </form>
    </dialog>
  );
}
