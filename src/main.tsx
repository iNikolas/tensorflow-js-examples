import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router";

import { RoutesProvider } from "@/components/providers/routes";

import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <HashRouter>
      <RoutesProvider />
    </HashRouter>
  </StrictMode>
);
