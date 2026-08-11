import { ALL_KNOWN_MODALITIES, modalityMeta } from "../../lib/modalities";
import "./EvidenceConstellation.css";

interface Props {
  weights: Record<string, number>;
  outlookLabel: string;
  outlookColorVar: string;
}

// An original EMAFRS visualization: the prediction sits at the center, and each modality
// that actually contributed real evidence orbits it -- distance from center and line
// weight both encode the REAL fusion weight (RAMA's exact softmax output, not an
// illustrative approximation; see src/emafrs/models/fusion.py). Modalities EMAFRS's
// taxonomy recognizes but that produced no evidence for THIS prediction are shown as
// faint, dashed ghost nodes at the rim -- so absence of evidence is visible, not hidden.
export function EvidenceConstellation({ weights, outlookLabel, outlookColorVar }: Props) {
  const present = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  const presentNames = new Set(present.map(([name]) => name));
  const absent = ALL_KNOWN_MODALITIES.filter((m) => !presentNames.has(m));

  const size = 280;
  const center = size / 2;
  const maxRadius = 105;
  const minRadius = 55;

  const totalSlots = present.length + absent.length;
  const angleFor = (i: number) => (2 * Math.PI * i) / totalSlots - Math.PI / 2;

  return (
    <div className="constellation-wrap">
      <svg viewBox={`0 0 ${size} ${size}`} width="100%" height="auto" role="img" aria-label="Evidence constellation">
        {present.map(([name, weight], i) => {
          const angle = angleFor(i);
          const r = minRadius + (maxRadius - minRadius) * (1 - weight); // stronger evidence = closer
          const x = center + r * Math.cos(angle);
          const y = center + r * Math.sin(angle);
          const meta = modalityMeta(name);
          return (
            <g key={name}>
              <line
                x1={center}
                y1={center}
                x2={x}
                y2={y}
                stroke={meta.color}
                strokeWidth={Math.max(1, weight * 6)}
                strokeOpacity={0.35 + weight * 0.5}
              />
              <circle cx={x} cy={y} r={10 + weight * 16} fill={meta.color} fillOpacity={0.85} />
              <text x={x} y={y + 10 + weight * 16 + 14} textAnchor="middle" className="constellation-label">
                {meta.short}
              </text>
              <text x={x} y={y + 4} textAnchor="middle" className="constellation-pct">
                {Math.round(weight * 100)}%
              </text>
            </g>
          );
        })}

        {absent.map((name, j) => {
          const angle = angleFor(present.length + j);
          const x = center + maxRadius * Math.cos(angle);
          const y = center + maxRadius * Math.sin(angle);
          const meta = modalityMeta(name);
          return (
            <g key={name} opacity={0.4}>
              <line x1={center} y1={center} x2={x} y2={y} stroke="var(--border-strong)" strokeWidth={1} strokeDasharray="3 4" />
              <circle cx={x} cy={y} r={8} fill="none" stroke="var(--border-strong)" strokeWidth={1.5} strokeDasharray="2 3" />
              <text x={x} y={y + 22} textAnchor="middle" className="constellation-label constellation-label-absent">
                {meta.short}
              </text>
              <text x={x} y={y + 34} textAnchor="middle" className="constellation-label constellation-label-absent">
                no evidence
              </text>
            </g>
          );
        })}

        <circle cx={center} cy={center} r={38} fill="var(--surface-raised)" stroke={outlookColorVar} strokeWidth={2} />
        <text x={center} y={center - 2} textAnchor="middle" className="constellation-center-label">
          Outlook
        </text>
        <text x={center} y={center + 14} textAnchor="middle" className="constellation-center-value" fill={outlookColorVar}>
          {outlookLabel}
        </text>
      </svg>
    </div>
  );
}
