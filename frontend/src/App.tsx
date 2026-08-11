import { Route, Routes } from "react-router-dom";
import "./App.css";
import "./styles/base.css";
import "./styles/consumer.css";
import { Shell } from "./layout/Shell";
import { OverviewPage } from "./pages/consumer/OverviewPage";
import { AssetProfilePage } from "./pages/consumer/AssetProfilePage";
import { AIIntelligencePage } from "./pages/consumer/AIIntelligencePage";
import { SentimentPage } from "./pages/consumer/SentimentPage";
import { ResearchHomePage } from "./pages/research/ResearchHomePage";
import { ModelPerformancePage } from "./pages/research/ModelPerformancePage";
import { CalibrationPage } from "./pages/research/CalibrationPage";
import { RegimeAnalysisPage } from "./pages/research/RegimeAnalysisPage";
import { ExplainabilityPage } from "./pages/research/ExplainabilityPage";
import { AffectiveComputingPage } from "./pages/research/AffectiveComputingPage";
import { DataProvenancePage } from "./pages/research/DataProvenancePage";
import { TrustworthinessAuditPage } from "./pages/research/TrustworthinessAuditPage";
import { MethodologyPage } from "./pages/research/MethodologyPage";
import { EvidenceConsolePage } from "./pages/research/EvidenceConsolePage";

function App() {
  return (
    <Routes>
      <Route element={<Shell mode="product" />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/asset/:ticker" element={<AssetProfilePage />} />
        <Route path="/intelligence" element={<AIIntelligencePage />} />
        <Route path="/sentiment" element={<SentimentPage />} />
      </Route>

      <Route element={<Shell mode="research" />}>
        <Route path="/research" element={<ResearchHomePage />} />
        <Route path="/research/performance" element={<ModelPerformancePage />} />
        <Route path="/research/calibration" element={<CalibrationPage />} />
        <Route path="/research/regime" element={<RegimeAnalysisPage />} />
        <Route path="/research/explainability" element={<ExplainabilityPage />} />
        <Route path="/research/affective" element={<AffectiveComputingPage />} />
        <Route path="/research/data" element={<DataProvenancePage />} />
        <Route path="/research/audit" element={<TrustworthinessAuditPage />} />
        <Route path="/research/methodology" element={<MethodologyPage />} />
        <Route path="/research/evidence/:ticker/:predictionId" element={<EvidenceConsolePage />} />
      </Route>
    </Routes>
  );
}

export default App;
