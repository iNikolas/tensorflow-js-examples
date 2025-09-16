import React from "react";
import { Navigate, Route, Routes, type RoutesProps } from "react-router";

import { MainLayout } from "@/components/containers/layouts/main-layout";

const ManipulatingImagesPage = React.lazy(
  () => import("@/pages/manipulating-images")
);

const SortingChaosChallengePage = React.lazy(
  () => import("@/pages/sorting-chaos-challenge")
);

const TicktackToePage = React.lazy(() => import("@/pages/tick-tack-toe"));

export function RoutesProvider({ ...props }: RoutesProps) {
  return (
    <Routes {...props}>
      <Route element={<MainLayout />}>
        <Route index element={<Navigate to="tick-tack-toe" />} />
        <Route
          path="manipulating-images"
          element={<ManipulatingImagesPage />}
        />
        <Route
          path="sorting-chaos-challenge"
          element={<SortingChaosChallengePage />}
        />
        <Route path="tick-tack-toe" element={<TicktackToePage />} />
      </Route>
    </Routes>
  );
}
