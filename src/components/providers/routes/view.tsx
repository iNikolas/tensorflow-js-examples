import React from "react";
import { Route, Routes, type RoutesProps } from "react-router";

import { MainLayout } from "@/components/containers/layouts/main-layout";

const HomePage = React.lazy(() => import("@/pages/home"));

export function RoutesProvider({ ...props }: RoutesProps) {
  return (
    <Routes {...props}>
      <Route element={<MainLayout />}>
        <Route path="/" element={<HomePage />} />
      </Route>
    </Routes>
  );
}
