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

const NonlinearRegressionPage = React.lazy(
  () => import("@/pages/nonlinear-regression")
);

const TitanicSurvivalProbabilityPage = React.lazy(
  () => import("@/pages/titanic-survival-probability")
);

const SortingHatPage = React.lazy(() => import("@/pages/sorting-hat"));

export function RoutesProvider({ ...props }: RoutesProps) {
  return (
    <Routes {...props}>
      <Route element={<MainLayout />}>
        <Route index element={<Navigate to="titanic-survival-probability" />} />
        <Route
          path="manipulating-images"
          element={<ManipulatingImagesPage />}
        />
        <Route
          path="sorting-chaos-challenge"
          element={<SortingChaosChallengePage />}
        />
        <Route path="inception-v3" element={<InceptionV3Page />} />
        <Route
          path="nonlinear-regression"
          element={<NonlinearRegressionPage />}
        />
        <Route
          path="titanic-survival-probability"
          element={<TitanicSurvivalProbabilityPage />}
        />
        <Route path="sorting-hat" element={<SortingHatPage />} />
      </Route>
    </Routes>
  );
}
