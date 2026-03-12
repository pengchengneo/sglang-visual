interface Props {
  title: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

export function SidebarSection({ title, open, onToggle, children }: Props) {
  return (
    <div className="sidebar-section">
      <button className="sidebar-section-header" onClick={onToggle}>
        <span className="sidebar-section-title">{title}</span>
        <span className={`sidebar-section-chevron ${open ? "open" : ""}`}>
          {open ? "\u25BE" : "\u25B8"}
        </span>
      </button>
      <div className={`sidebar-section-body ${open ? "open" : ""}`}>
        <div className="sidebar-section-content">{children}</div>
      </div>
    </div>
  );
}
