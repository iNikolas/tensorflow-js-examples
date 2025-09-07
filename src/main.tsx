import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";

import { RoutesProvider } from "@/components/providers/routes";

import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <RoutesProvider />
    </BrowserRouter>
  </StrictMode>
);
