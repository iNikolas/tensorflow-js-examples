import React from "react";
import { Outlet } from "react-router";

export function MainLayout() {
  return (
    <div className="flex max-h-screen flex-col">
      <header />
      <div className="flex flex-1 flex-col overflow-auto">
        <main className="flex flex-1 flex-col justify-center items-center">
          <React.Suspense
            fallback={<div className="loading loading-bars text-primary" />}
          >
            <Outlet />
          </React.Suspense>
        </main>
        <footer />
      </div>
    </div>
  );
}
