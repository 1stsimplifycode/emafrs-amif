import { useState, type FormEvent } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import "./Shell.css";

const PRODUCT_NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/intelligence", label: "AI Intelligence" },
  { to: "/sentiment", label: "Sentiment" },
];

const RESEARCH_NAV = [
  { to: "/research", label: "Overview", end: true },
  { to: "/research/performance", label: "Model Performance" },
  { to: "/research/calibration", label: "Calibration" },
  { to: "/research/regime", label: "Regime Analysis" },
  { to: "/research/explainability", label: "Explainability" },
  { to: "/research/affective", label: "Affective Computing" },
  { to: "/research/data", label: "Data Provenance" },
  { to: "/research/audit", label: "Trustworthiness Audit" },
  { to: "/research/methodology", label: "Methodology" },
];

export function Shell({ mode }: { mode: "product" | "research" }) {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");

  function submitSearch(e: FormEvent) {
    e.preventDefault();
    const t = query.trim().toUpperCase();
    if (t) {
      navigate(`/asset/${t}`);
      setQuery("");
    }
  }

  const navItems = mode === "product" ? PRODUCT_NAV : RESEARCH_NAV;

  return (
    <div className="shell">
      <header className="shell-topbar">
        <div className="shell-topbar-inner">
          <div className="shell-brand">
            <NavLink to="/" className="brand-mark">
              <span className="brand-glyph" aria-hidden="true">
                <span className="brand-glyph-dot" />
              </span>
              <span className="brand-word">EMAFRS</span>
            </NavLink>
            <div className="mode-switch" role="tablist" aria-label="Product mode">
              <NavLink to="/" role="tab" className={`mode-tab ${mode === "product" ? "mode-tab-active" : ""}`}>
                Product
              </NavLink>
              <NavLink to="/research" role="tab" className={`mode-tab ${mode === "research" ? "mode-tab-active" : ""}`}>
                Research &amp; Trust
              </NavLink>
            </div>
          </div>

          {mode === "product" && (
            <form className="shell-search" onSubmit={submitSearch}>
              <input
                className="input shell-search-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search a ticker (e.g. NVDA)"
                aria-label="Search a ticker"
              />
              <button className="btn btn-primary" type="submit">
                Explore
              </button>
            </form>
          )}
        </div>
        <nav className="shell-subnav" aria-label={mode === "product" ? "Product navigation" : "Research navigation"}>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `subnav-link ${isActive ? "subnav-link-active" : ""}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="shell-main">
        <Outlet />
      </main>

      <footer className="shell-footer">
        <p className="text-faint">
          EMAFRS / AMIF — a research reconstruction. Every number shown comes from a real,
          live-trained model or a real API call, cross-linked to the underlying evidence
          in the Research &amp; Trust workspace. Nothing here is a placeholder.
        </p>
      </footer>
    </div>
  );
}
