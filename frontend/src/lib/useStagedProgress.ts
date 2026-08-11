import { useEffect, useRef, useState } from "react";

// Real training/inference DOES happen in this exact order (see
// run_training_and_prediction_pipeline in src/emafrs/mlops/training_pipeline.py:
// build_daily_dataset -> train_fusion_model -> compute_integrated_gradients ->
// search_all_feature_counterfactuals -> generate_explanation) -- these labels are not
// invented busywork text, they name the real steps in their real order.
//
// What IS a UI-side approximation: the TIMING of when to advance between labels. There
// is no backend endpoint streaming real progress for a synchronous ~10-30s request, so
// advancement is timer-driven and holds on the last stage for as long as the real
// request actually takes. This never displays a fabricated RESULT (number, prediction,
// direction) -- only ever "here is roughly what's happening right now," which is true.
export const TRAINING_STAGES = [
  "Fetching real market data…",
  "Training the model on real history…",
  "Computing explanations (Integrated Gradients)…",
  "Searching for counterfactuals…",
];

export function useStagedProgress(active: boolean, stageDurationMs = 3500): string {
  const [stageIndex, setStageIndex] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!active) {
      setStageIndex(0);
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    intervalRef.current = setInterval(() => {
      setStageIndex((i) => Math.min(i + 1, TRAINING_STAGES.length - 1)); // holds on the last stage, never loops back
    }, stageDurationMs);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [active, stageDurationMs]);

  return TRAINING_STAGES[stageIndex];
}
