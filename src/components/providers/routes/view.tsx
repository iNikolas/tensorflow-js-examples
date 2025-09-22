import React from "react";
import { Navigate, Route, Routes, type RoutesProps } from "react-router";

import { MainLayout } from "@/components/containers/layouts/main-layout";

const ManipulatingImagesPage = React.lazy(
  () => import("@/pages/manipulating-images")
);

const SortingChaosChallengePage = React.lazy(
  () => import("@/pages/sorting-chaos-challenge")
);

const InceptionV3Page = React.lazy(() => import("@/pages/inception-v3"));

const HomePage = React.lazy(() => import("@/pages/home"));

export function RoutesProvider({ ...props }: RoutesProps) {
  return (
    <Routes {...props}>
      <Route element={<MainLayout />}>
        <Route index element={<Navigate to="home" />} />
        <Route
          path="manipulating-images"
          element={<ManipulatingImagesPage />}
        />
        <Route
          path="sorting-chaos-challenge"
          element={<SortingChaosChallengePage />}
        />
        <Route path="inception-v3" element={<InceptionV3Page />} />
        <Route path="home" element={<HomePage />} />
      </Route>
    </Routes>
  );
}
