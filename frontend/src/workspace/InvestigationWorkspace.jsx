import { useEffect, useMemo, useState } from "react";



import {

  useNavigate,

  useParams,

} from "react-router-dom";



import {
  getAlert,
  getCase,
  getCases,
  getInvestigationGraph,
  getTransactionById,
  submitCaseReview,
} from "../services/api";
import {
  formatScorePercentage,
  getScorePercentage,
  mergeTransactionRiskScores,
} from "../utils/riskScores";





import TransactionNetwork from "./TransactionNetwork";



function formatAmount(amount) {

  return new Intl.NumberFormat("en-IN", {

    style: "currency",

    currency: "INR",

    maximumFractionDigits: 2,

  }).format(amount);

}



function formatPercentage(value) {
  return formatScorePercentage(value);
}



function formatDate(timestamp) {

  if (!timestamp) {

    return "—";

  }



  const date = new Date(

    String(timestamp).replace(" ", "T")

  );



  if (Number.isNaN(date.getTime())) {

    return timestamp;

  }



  return date.toLocaleString("en-IN", {

    day: "2-digit",

    month: "short",

    year: "numeric",

    hour: "2-digit",

    minute: "2-digit",

  });

}



function formatPattern(value) {

  if (!value) {

    return "Risk signal";

  }



  return String(value)

    .replaceAll("_", " ")

    .replace(/\b\w/g, (letter) =>

      letter.toUpperCase()

    );

}



function RiskBadge({ category }) {

  const riskCategory = category || "Not scored";

  const normalized =

    String(riskCategory).toLowerCase();



  return (

    <span

      className={`workspace-risk-badge workspace-risk-${normalized}`}

    >

      <span className="workspace-risk-dot" />



      {riskCategory}

    </span>

  );

}



function MetricCard({

  label,

  value,

  description,

  className = "",

}) {

  return (

    <div

      className={`workspace-metric-card ${className}`}

    >

      <div className="workspace-metric-label">

        {label}

      </div>



      <div className="workspace-metric-value">

        {value}

      </div>



      <div className="workspace-metric-description">

        {description}

      </div>

    </div>

  );

}



function WorkspaceLoading() {

  return (

    <section className="page-content">

      <div className="workspace-loading">

        <div className="loading-spinner" />



        <div>

          <div className="loading-title">

            Loading investigation

          </div>



          <div className="loading-description">

            Retrieving case and graph evidence...

          </div>

        </div>

      </div>

    </section>

  );

}



function WorkspaceError({ message }) {

  const navigate = useNavigate();



  return (

    <section className="page-content">

      <div className="workspace-error">

        <div className="error-title">

          Investigation unavailable

        </div>



        <div className="error-description">

          AegisGraph could not load the investigation

          data.

        </div>



        <div className="error-details">

          {message}

        </div>



        <button

          type="button"

          className="back-button"

          onClick={() => navigate("/alerts")}

        >

          ← Back to Alerts

        </button>

      </div>

    </section>

  );

}



function GraphNode({

  label,

  type,

  x,

  y,

  active = false,

}) {

  return (

    <div

      className={`workspace-graph-node workspace-graph-node-${type} ${

        active

          ? "workspace-graph-node-active"

          : ""

      }`}

      style={{

        left: `${x}%`,

        top: `${y}%`,

      }}

    >

      <div className="workspace-node-circle">

        {type === "sender"

          ? "S"

          : type === "receiver"

            ? "R"

            : "A"}

      </div>



      <div className="workspace-node-label">

        {label}

      </div>

    </div>

  );

}



function GraphEdge({

  className = "",

}) {

  return (

    <div

      className={`workspace-graph-edge ${className}`}

    />

  );

}



function InvestigationGraph({

  alert,

  caseData,

  graphData 

}) {

  const senderDegree =

    Number(

      caseData?.sender_total_degree || 0

    );



  const receiverDegree =

    Number(

      caseData?.receiver_total_degree || 0

    );



  const relatedTransactions =

    Number(

      caseData?.related_transaction_count ||

        0

    );



  return (

    <div className="workspace-graph-card">

      <div className="workspace-graph-header">

        <div>

          <h2>Transaction Network</h2>



          <p>

            Simplified view of the accounts involved

            in this investigation.

          </p>

        </div>



        <div className="workspace-graph-controls">

          <button type="button">−</button>

          <button type="button">+</button>

          <button type="button">Reset</button>

        </div>

      </div>



      <div className="workspace-graph-canvas">

        <TransactionNetwork

          graphData={graphData}

          senderAccount={alert.sender_account}

          receiverAccount={alert.receiver_account}

          transactionId={alert.transaction_id}

          amount={alert.amount}

          senderConnections={caseData.sender_total_degree}

          receiverConnections={caseData.receiver_total_degree}

          relatedTransactions={caseData.related_transaction_count}

        />

      </div>



      <div className="workspace-graph-legend">

        <div>

          <span className="legend-dot sender" />

          Sender account

        </div>



        <div>

          <span className="legend-dot transaction" />

          Investigated transaction

        </div>



        <div>

          <span className="legend-dot receiver" />

          Receiver account

        </div>

      </div>

    </div>

  );

}



function SignalBar({

  label,

  value,

  description,

  type,

}) {

  const scorePercentage = getScorePercentage(value);
  const percentage = scorePercentage === null
    ? null
    : Math.max(0, Math.min(100, scorePercentage));



  return (

    <div className="workspace-signal">

      <div className="workspace-signal-top">

        <span>{label}</span>



        <strong>

          {percentage === null
            ? "Not scored"
            : `${percentage.toFixed(2)}%`}

        </strong>

      </div>



      <div className="workspace-signal-track">

        <div

          className={`workspace-signal-fill ${type}`}

          style={{

            width: percentage === null ? "0%" : `${percentage}%`,

          }}

        />

      </div>



      <div className="workspace-signal-description">

        {description}

      </div>

    </div>

  );

}



function RuleSignal({

  label,

  triggered,

}) {

  return (

    <div className="workspace-rule-row">

      <div

        className={

          triggered

            ? "workspace-rule-icon triggered"

            : "workspace-rule-icon"

        }

      >

        {triggered ? "✓" : "—"}

      </div>



      <span>{label}</span>



      <strong

        className={

          triggered

            ? "workspace-rule-triggered"

            : ""

        }

      >

        {triggered

          ? "Triggered"

          : "Not triggered"}

      </strong>

    </div>

  );

}



function TimelineItem({

  number,

  title,

  description,

  active = false,

}) {

  return (

    <div className="workspace-timeline-item">

      <div

        className={

          active

            ? "workspace-timeline-number active"

            : "workspace-timeline-number"

        }

      >

        {number}

      </div>



      <div>

        <div className="workspace-timeline-title">

          {title}

        </div>



        <div className="workspace-timeline-description">

          {description}

        </div>

      </div>

    </div>

  );

}



export default function InvestigationWorkspace() {

  const { alertId } = useParams();



  const navigate = useNavigate();



  const [alert, setAlert] = useState(null);

  const [caseData, setCaseData] = useState(null);

  const [graphData, setGraphData] = useState(null);

  const [loading, setLoading] = useState(true);

  const [error, setError] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [reviewComments, setReviewComments] = useState("");
  const [savingReview, setSavingReview] = useState(false);
  const [reviewMessage, setReviewMessage] = useState("");
  const [reviewError, setReviewError] = useState("");



  useEffect(() => {

    let mounted = true;



    async function loadWorkspace() {

      try {

        setLoading(true);

        setError("");



        const alertData =

          await getAlert(alertId);

        const transaction = await getTransactionById(
          alertData.transaction_id
        );
        const riskEvidence = mergeTransactionRiskScores(
          alertData,
          transaction
        );

        if (!mounted) {

          return;

        }



        setAlert(riskEvidence);

        const graphData = await getInvestigationGraph(alertData.alert_id);

        setGraphData(graphData);



        const casesResponse =

          await getCases();



        if (!mounted) {

          return;

        }



        const cases =

          casesResponse.cases || [];



        const matchingCase =

          cases.find(

            (item) =>

              item.alert_id ===

              alertData.alert_id

          );



        if (!matchingCase) {

          throw new Error(

            `No investigation case was found for ${alertData.alert_id}.`

          );

        }



        const detailedCase =

          await getCase(

            matchingCase.case_id

          );



        if (!mounted) {

          return;

        }



        setCaseData(
          mergeTransactionRiskScores(
            detailedCase,
            transaction || alertData
          )
        );

      } catch (err) {

        if (mounted) {

          setError(

            err instanceof Error

              ? err.message

              : "Unknown API error"

          );

        }

      } finally {

        if (mounted) {

          setLoading(false);

        }

      }

    }



    if (alertId) {

      loadWorkspace();

    } else {

      setError(

        "No alert ID was provided for this investigation."

      );

      setLoading(false);

    }



    return () => {

      mounted = false;

    };

  }, [alertId]);



  
  async function handleReviewAction(action) {
    if (savingReview) return;

    if (!caseData?.case_id) {
      setReviewError("No case is loaded for this investigation.");
      return;
    }

    if (!reviewer.trim()) {
      setReviewError("Enter the investigator name before submitting.");
      return;
    }

    setSavingReview(true);
    setReviewError("");
    setReviewMessage("");

    try {
      const result = await submitCaseReview(caseData.case_id, {
        action,
        reviewer: reviewer.trim(),
        comments: reviewComments.trim(),
      });

      // Refresh the case using the same case ID.
      const updatedCase = await getCase(caseData.case_id);
      setCaseData(updatedCase);

      setReviewComments("");

      const actualStatus =
        updatedCase.case_status ??
        result.target_status ??
        "Updated";

      setReviewMessage(
        `Review submitted successfully. Case status: ${actualStatus}.`
      );
    } catch (err) {
      setReviewError(
        err.response?.data?.detail ||
          err.message ||
          "Failed to submit the review."
      );
    } finally {
      setSavingReview(false);
    }
  }


  const triggeredRules = useMemo(() => {

    if (!alert) {

      return [];

    }



    return [

      {

        label: "Large Transaction",

        triggered: Boolean(

          alert.large_transaction

        ),

      },

      {

        label: "Structuring",

        triggered: Boolean(

          alert.structuring

        ),

      },

      {

        label: "Rapid Movement",

        triggered: Boolean(

          alert.rapid_movement

        ),

      },

      {

        label: "Fan-out",

        triggered: Boolean(

          alert.fan_out

        ),

      },

      {

        label: "Circular Movement",

        triggered: Boolean(

          alert.circular_movement

        ),

      },

    ];

  }, [alert]);



  if (loading) {

    return <WorkspaceLoading />;

  }



  if (error || !alert || !caseData) {

    return (

      <WorkspaceError

        message={

          error ||

          "Investigation data is incomplete."

        }

      />

    );

  }



  return (

    <section className="page-content workspace-page">

      <div className="workspace-back">

        <button

          type="button"

          className="back-button"

          onClick={() =>

            navigate(

              `/alerts/${alert.alert_id}`

            )

          }

        >

          ← Back to Alert

        </button>

      </div>



      <div className="workspace-header">

        <div>

          <div className="page-eyebrow">

            INVESTIGATION WORKSPACE

          </div>



          <div className="workspace-title-row">

            <h1>

              {caseData.case_id}

            </h1>



            <RiskBadge

              category={

                caseData.risk_category

              }

            />

          </div>



          <p>

            Investigating transaction{" "}

            <strong>

              {caseData.transaction_id}

            </strong>{" "}

            from alert{" "}

            <strong>

              {caseData.alert_id}

            </strong>

            .

          </p>

        </div>



        <div className="workspace-header-status">

          <div>

            <span>REVIEW</span>



            <strong>

              {caseData.review_status ||

                "PENDING_REVIEW"}

            </strong>

          </div>



          <div>

            <span>DECISION</span>



            <strong>

              {caseData.investigator_decision ||

                "UNDECIDED"}

            </strong>

          </div>

        </div>

      </div>



      <div className="workspace-summary">

        <MetricCard

          label="Overall Risk"

          value={formatPercentage(

            caseData.risk_score

          )}

          description="Final fused risk assessment."

          className="workspace-risk-card"

        />



        <MetricCard

          label="Transaction"

          value={formatAmount(

            caseData.amount ??

              alert.amount

          )}

          description={`${alert.sender_account} → ${alert.receiver_account}`}

        />



        <MetricCard

          label="Related Activity"

          value={

            caseData.related_transaction_count ??

            0

          }

          description="Related transactions identified in the graph."

        />



        <MetricCard

          label="Triggered Rules"

          value={

            triggeredRules.filter(

              (rule) => rule.triggered

            ).length

          }

          description="AML rules contributing to the alert."

        />

      </div>



      <div className="workspace-main-grid">

        <div className="workspace-left">

          <InvestigationGraph

            alert={alert}

            caseData={caseData}

            graphData={graphData}

          />



          <section className="workspace-panel">

            <div className="workspace-panel-heading">

              <div>

                <h2>

                  Investigation Timeline

                </h2>



                <p>

                  How AegisGraph reached the current

                  investigation state.

                </p>

              </div>

            </div>



            <div className="workspace-timeline">

              <TimelineItem

                number="1"

                title="Transaction received"

                description="The transaction entered the detection pipeline."

              />



              <TimelineItem

                number="2"

                title="Features calculated"

                description="Transaction and account-level signals were prepared."

              />



              <TimelineItem

                number="3"

                title="Machine learning evaluated risk"

                description="Random Forest and Temporal Graph Model produced risk probabilities."

              />



              <TimelineItem

                number="4"

                title="Graph relationships analyzed"

                description="Account connectivity and related activity were evaluated."

              />



              <TimelineItem

                number="5"

                title="AML rules evaluated"

                description="Rule-based suspicious activity patterns were checked."

              />



              <TimelineItem

                number="6"

                title="Risk signals combined"

                description="ML and rule signals were fused into the final risk score."

              />



              <TimelineItem

                number="7"

                title="Alert generated"

                description="The transaction was added to the investigation queue."

                active

              />



              <TimelineItem

                number="8"

                title="Case created"

                description="The alert was converted into an investigation case."

                active

              />

            </div>

          </section>

        </div>



        <div className="workspace-right">

          <section className="workspace-panel">

            <div className="workspace-panel-heading">

              <div>

                <h2>

                  Risk Assessment

                </h2>



                <p>

                  Evidence supporting the current risk

                  classification.

                </p>

              </div>

            </div>



            <div className="workspace-signals">

              <SignalBar

                label="Random Forest"

                value={

                  caseData.rf_probability

                }

                description="Transaction-level machine learning probability."

                type="blue"

              />



              <SignalBar

                label="Temporal Graph Network (TGN)"

                value={

                  caseData.tgn_probability

                }

                description="Temporal account-network risk probability."

                type="purple"

              />

              <SignalBar
                label="Temporal Heterogeneous Graph Neural Network (THGNN)"
                value={caseData.thgnn_probability}
                description="Probability assigned by the temporal heterogeneous graph model."
                type="teal"
              />

              <SignalBar

                label="AML Rule Score"

                value={

                  caseData.rule_score

                }

                description="Combined contribution from AML detection rules."

                type="orange"

              />

            </div>

          </section>



          <section className="workspace-panel">

            <div className="workspace-panel-heading">

              <div>

                <h2>

                  Detection Rules

                </h2>



                <p>

                  Rule signals evaluated for this

                  transaction.

                </p>

              </div>

            </div>



            <div className="workspace-rules">

              {triggeredRules.map(

                (rule) => (

                  <RuleSignal

                    key={rule.label}

                    label={rule.label}

                    triggered={

                      rule.triggered

                    }

                  />

                )

              )}

            </div>

          </section>



          <section className="workspace-panel">

            <div className="workspace-panel-heading">

              <div>

                <h2>

                  Graph Evidence

                </h2>



                <p>

                  Account-level network context.

                </p>

              </div>

            </div>



            <div className="workspace-evidence-list">

              <div>

                <span>

                  Sender connections

                </span>



                <strong>

                  {

                    caseData.sender_total_degree

                  }

                </strong>

              </div>



              <div>

                <span>

                  Receiver connections

                </span>



                <strong>

                  {

                    caseData.receiver_total_degree

                  }

                </strong>

              </div>



              <div>

                <span>

                  Related transactions

                </span>



                <strong>

                  {

                    caseData.related_transaction_count

                  }

                </strong>

              </div>



              <div>

                <span>

                  Detected pattern

                </span>



                <strong>

                  {formatPattern(

                    alert.typology

                  )}

                </strong>

              </div>

            </div>

          </section>



          <section className="workspace-conclusion">

            <div className="workspace-conclusion-label">

              CURRENT ASSESSMENT

            </div>



            <h2>

              {caseData.risk_category} RISK

            </h2>



            <p>

              This transaction requires analyst

              review based on combined machine

              learning, graph and rule evidence.

            </p>

          </section>

        </div>

      </div>



      <section className="workspace-decision">
        <div>
          <div className="workspace-panel-heading">
            <div>
              <h2>Investigator Decision</h2>
              <p>Record your review and update the case workflow.</p>
            </div>
          </div>
        </div>

        <div className="workspace-decision-note">
          <strong>Review status:</strong>{" "}
          {caseData.review_status || "PENDING_REVIEW"}
          {" · "}
          <strong>Decision:</strong>{" "}
          {caseData.investigator_decision || "UNDECIDED"}
          {" · "}
          <strong>Case status:</strong>{" "}
          {caseData.case_status || "OPEN"}
        </div>

        <div className="workspace-decision-form">
          <label htmlFor="reviewer-name">Investigator name</label>
          <input
            id="reviewer-name"
            type="text"
            value={reviewer}
            onChange={(event) => setReviewer(event.target.value)}
            placeholder="Enter investigator name"
            disabled={savingReview}
          />

          <label htmlFor="review-comments">Review comments</label>
          <textarea
            id="review-comments"
            value={reviewComments}
            onChange={(event) => setReviewComments(event.target.value)}
            placeholder="Document the evidence and rationale for your action"
            rows={4}
            disabled={savingReview}
          />
        </div>

        {reviewError && (
          <p role="alert" className="workspace-review-error">
            {reviewError}
          </p>
        )}

        {reviewMessage && (
          <p role="status" className="workspace-review-success">
            {reviewMessage}
          </p>
        )}

        <div className="workspace-decision-actions">
          <button
            type="button"
            className="decision-button"
            disabled={savingReview}
            onClick={() => handleReviewAction("MARK_AS_LEGITIMATE")}
          >
            Mark Legitimate
          </button>
          <button
            type="button"
            className="decision-button"
            disabled={savingReview}
            onClick={() => handleReviewAction("MARK_AS_FRAUD")}
          >
            Mark as Fraud
          </button>
          <button
            type="button"
            className="decision-button"
            disabled={savingReview}
            onClick={() => handleReviewAction("REQUEST_MORE_INFORMATION")}
          >
            Request Information
          </button>
          <button
            type="button"
            className="decision-button primary"
            disabled={savingReview}
            onClick={() => handleReviewAction("ESCALATE")}
          >
            Escalate
          </button>
          <button
            type="button"
            className="decision-button"
            disabled={savingReview}
            onClick={() => handleReviewAction("CLOSE")}
          >
            Close Case
          </button>
        </div>

        {savingReview && <p role="status">Submitting review…</p>}
      </section>

    </section>

  );

}