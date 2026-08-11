import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Shell } from "./Shell";

function renderShell(mode: "product" | "research", path = "/") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route element={<Shell mode={mode} />}>
          <Route path="/" element={<div>overview content</div>} />
          <Route path="/research" element={<div>research content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>
  );
}

describe("Shell", () => {
  it("shows the product navigation and search bar in product mode", () => {
    renderShell("product");
    expect(screen.getByText("AI Intelligence")).toBeInTheDocument();
    expect(screen.getByText("Sentiment")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Search a ticker/)).toBeInTheDocument();
  });

  it("shows the research navigation and hides the ticker search in research mode", () => {
    renderShell("research", "/research");
    expect(screen.getByText("Calibration")).toBeInTheDocument();
    expect(screen.getByText("Trustworthiness Audit")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText(/Search a ticker/)).not.toBeInTheDocument();
  });

  it("renders the EMAFRS brand mark", () => {
    renderShell("product");
    expect(screen.getByText("EMAFRS")).toBeInTheDocument();
  });

  it("renders the routed page content via the Outlet", () => {
    renderShell("product");
    expect(screen.getByText("overview content")).toBeInTheDocument();
  });
});
