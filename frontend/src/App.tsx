import { useState, useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { AppProvider } from "./contexts/AppContext";
import TopNav from "./components/navigation/TopNav";
import { Sidebar } from "./components/sidebar/Sidebar";
import ComputePlanePage from "./pages/ComputePlanePage";
import ControlPlanePage from "./pages/ControlPlanePage";
import KVCachePage from "./pages/KVCachePage";
import SchedulingPage from "./pages/SchedulingPage";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./App.css";

type Theme = "light" | "dark" | "system";

function getInitialTheme(): Theme {
  const stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") return stored;
  return "system";
}

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", theme);
  }
}

function useTheme() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    applyTheme(theme);
    if (theme === "system") {
      localStorage.removeItem("theme");
    } else {
      localStorage.setItem("theme", theme);
    }
  }, [theme]);

  const toggle = () => {
    setTheme((prev) => {
      if (prev === "system") {
        const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        return isDark ? "light" : "dark";
      }
      return prev === "light" ? "dark" : "light";
    });
  };

  const isDark =
    theme === "dark" ||
    (theme === "system" &&
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);

  return { theme, isDark, toggle };
}

function AppContent() {
  const { isDark, toggle: toggleTheme } = useTheme();

  return (
    <div className="app">
      <TopNav />

      <div className="app-body">
        <Sidebar />

        <main className="main-content">
          <ErrorBoundary fallbackTitle="Page error">
            <Routes>
              <Route path="/compute" element={<ComputePlanePage />} />
              <Route path="/control" element={<ControlPlanePage />} />
              <Route path="/kv-cache" element={<KVCachePage />} />
              <Route path="/scheduling" element={<SchedulingPage />} />
              <Route path="/" element={<Navigate to="/compute" replace />} />
            </Routes>
          </ErrorBoundary>
        </main>
      </div>

      <button
        className="theme-toggle"
        onClick={toggleTheme}
        title="Toggle theme"
        aria-label="Toggle theme"
      >
        {isDark ? "☀️" : "🌙"}
      </button>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
