import "./shared.css";

interface Props {
  labelA: string;
  labelB: string;
  active: "a" | "b";
  onChange: (mode: "a" | "b") => void;
}

export default function ComparisonToggle({ labelA, labelB, active, onChange }: Props) {
  return (
    <div className="comparison-toggle">
      <button className={`comparison-btn${active === "a" ? " active" : ""}`} onClick={() => onChange("a")}>{labelA}</button>
      <button className={`comparison-btn${active === "b" ? " active" : ""}`} onClick={() => onChange("b")}>{labelB}</button>
    </div>
  );
}
