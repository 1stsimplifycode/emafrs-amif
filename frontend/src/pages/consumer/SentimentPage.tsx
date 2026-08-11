import { Link } from "react-router-dom";
import "./SentimentPage.css";

// Deliberately careful naming throughout this page: "Market Sentiment" / "News
// Sentiment" / "Affective Context," never "emotion recognition" -- and no fabricated
// V-A-D or correlation numbers. What's real is presented as real; what isn't validated
// or wired into live predictions is stated plainly, not implied.
export function SentimentPage() {
  return (
    <div className="sentiment-page">
      <div className="panel-section">
        <span className="eyebrow">Market &amp; News Sentiment</span>
        <h1 className="display-xl">The affective side of EMAFRS</h1>
        <p className="text-muted" style={{ maxWidth: 640, marginTop: 8 }}>
          EMAFRS reads financial text — news and commentary — for sentiment and affective
          context. This page describes exactly what that component does, what's been
          validated, and, honestly, what isn't part of your live predictions yet.
        </p>
      </div>

      <section className="panel-section">
        <div className="section-head">
          <h2 className="display-md">Market sentiment analysis</h2>
        </div>
        <div className="card sentiment-card">
          <p>
            EMAFRS scores financial text with FinBERT, a language model trained
            specifically on financial sentiment (not a generic movie-review or
            social-media sentiment model). It reports a sentence's polarity — positive,
            negative, or neutral about the market — not a general emotional state.
          </p>
          <div className="sentiment-stat-row">
            <div>
              <span className="sentiment-stat-value numeric">89.2%</span>
              <span className="text-faint">accuracy on human-labeled financial sentences</span>
            </div>
          </div>
          <p className="text-faint sentiment-caveat">
            Real, measured result — but on Financial PhraseBank, the same public dataset
            this model was originally trained on, so this is likely an optimistic
            upper bound, not a clean out-of-sample number. Full method and the caveat, in
            full: <Link to="/research/affective">Research → Affective Computing</Link>.
          </p>
        </div>
      </section>

      <section className="panel-section">
        <div className="section-head">
          <h2 className="display-md">Affective context</h2>
        </div>
        <div className="card sentiment-card">
          <p>
            Beyond positive/negative sentiment, EMAFRS can also score financial text
            across seven contextual categories drawn from how financial communication is
            actually written: confidence, uncertainty, optimism, fear, caution,
            assertiveness, and evasiveness.
          </p>
          <p className="text-faint sentiment-caveat">
            <strong>Honestly labeled, not overclaimed:</strong> this is NOT validated
            human emotion recognition. No labeled ground-truth dataset exists for this
            specific 7-category taxonomy, so it has only been checked for directional
            sanity (a fear-coded sentence scores higher on "fear" than a confidence-coded
            one), never for accuracy. Treat it as a structured signal, not a proven
            measurement.
          </p>
        </div>
      </section>

      <section className="panel-section">
        <div className="section-head">
          <h2 className="display-md">Is this in your live predictions?</h2>
        </div>
        <div className="card sentiment-card sentiment-status-card">
          <p>
            <strong>Not currently.</strong> Every real attempt to feed live news/text data
            into a trained prediction has run into the same wall: not enough real,
            time-aligned text data has accumulated yet for the model to learn from it
            reliably. That's a real, measured data-volume limitation — not a broken
            pipeline, and not evidence that sentiment "doesn't help." The full technical
            account, including two real pilot runs and a fix that resolved part of the
            problem, is in <Link to="/research/affective">Research → Affective Computing</Link>.
          </p>
        </div>
      </section>
    </div>
  );
}
