export type Plane = "control" | "compute";

interface Props {
  active: Plane;
  onChange: (plane: Plane) => void;
}

const TABS: { id: Plane; label: string }[] = [
  { id: "control", label: "Control Plane" },
  { id: "compute", label: "Compute Plane" },
];

export function PlaneTabBar({ active, onChange }: Props) {
  return (
    <div className="tab-bar">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          className={`tab-btn${active === tab.id ? " active" : ""}`}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
