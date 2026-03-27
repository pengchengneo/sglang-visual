import { NavLink } from "react-router-dom";
import "./TopNav.css";

const TABS = [
  { to: "/compute", label: "Compute" },
  { to: "/control", label: "Control" },
  { to: "/kv-cache", label: "KV Cache" },
  { to: "/scheduling", label: "Scheduling" },
] as const;

export default function TopNav() {
  return (
    <nav className="top-nav">
      <div className="top-nav-brand">SGLang Visual</div>
      <div className="top-nav-tabs">
        {TABS.map((tab) => (
          <NavLink
            key={tab.to}
            to={tab.to}
            className={({ isActive }) =>
              `top-nav-tab${isActive ? " active" : ""}`
            }
          >
            {tab.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
