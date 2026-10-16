import React, { useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

function formatAmount(amount, currency = "") {
  if (amount === undefined || amount === null || Number.isNaN(Number(amount))) {
    return "";
  }

  return `${currency ? `${currency} ` : ""}${Number(amount).toLocaleString(
    "en-IN",
    {
      maximumFractionDigits: 2,
    }
  )}`;
}

function shortLabel(node) {
  if (!node) return "";

  if (node.type === "transaction") {
    return node.transaction_id || node.label || node.id;
  }

  return node.label || node.id;
}

function nodeColor(node) {
  if (node.central && node.type === "transaction") {
    return "#dc2626";
  }

  if (node.central && node.type === "sender") {
    return "#059669";
  }

  if (node.central && node.type === "receiver") {
    return "#2563eb";
  }

  if (node.type === "transaction") {
    return "#f59e0b";
  }

  return "#64748b";
}

function nodeRadius(node) {
  if (node.central && node.type === "transaction") {
    return 14;
  }

  if (node.central) {
    return 11;
  }

  if (node.type === "transaction") {
    return 7;
  }

  return 8;
}

function NodeDetails({ node }) {
  if (!node) return null;

  return (
    <div className="network-node-details">
      <div className="network-node-details-title">
        {shortLabel(node)}
      </div>

      <div className="network-node-details-row">
        <span>Type</span>
        <strong>{node.type}</strong>
      </div>

      {node.transaction_id && (
        <div className="network-node-details-row">
          <span>Transaction</span>
          <strong>{node.transaction_id}</strong>
        </div>
      )}

      {node.amount !== undefined && (
        <div className="network-node-details-row">
          <span>Amount</span>
          <strong>
            {formatAmount(node.amount, node.currency)}
          </strong>
        </div>
      )}

      {node.timestamp && (
        <div className="network-node-details-row">
          <span>Time</span>
          <strong>
            {new Date(node.timestamp).toLocaleString()}
          </strong>
        </div>
      )}

      {node.sender_account && (
        <div className="network-node-details-row">
          <span>From</span>
          <strong>{node.sender_account}</strong>
        </div>
      )}

      {node.receiver_account && (
        <div className="network-node-details-row">
          <span>To</span>
          <strong>{node.receiver_account}</strong>
        </div>
      )}

      {node.transaction_type && (
        <div className="network-node-details-row">
          <span>Type</span>
          <strong>{node.transaction_type}</strong>
        </div>
      )}

      {node.channel && (
        <div className="network-node-details-row">
          <span>Channel</span>
          <strong>{node.channel}</strong>
        </div>
      )}

      {node.connections !== undefined && (
        <div className="network-node-details-row">
          <span>Connections</span>
          <strong>{node.connections}</strong>
        </div>
      )}
    </div>
  );
}

export default function TransactionNetwork({
  graphData,
  senderAccount,
  receiverAccount,
  transactionId,
  amount,
}) {
  const graphRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);

  const graph = useMemo(() => {
    if (!graphData?.network) {
      return {
        nodes: [],
        links: [],
      };
    }

    const rawNodes = Array.isArray(graphData.network.nodes)
      ? graphData.network.nodes
      : [];

    const rawLinks = Array.isArray(graphData.network.links)
      ? graphData.network.links
      : [];

    const nodes = rawNodes.map((node) => ({
      ...node,
      id: String(node.id),
    }));

    const nodeIds = new Set(nodes.map((node) => node.id));

    const links = rawLinks
      .filter((link) => {
        const source = String(
          typeof link.source === "object"
            ? link.source.id
            : link.source
        );

        const target = String(
          typeof link.target === "object"
            ? link.target.id
            : link.target
        );

        return nodeIds.has(source) && nodeIds.has(target);
      })
      .map((link) => ({
        ...link,
        source:
          typeof link.source === "object"
            ? link.source.id
            : String(link.source),
        target:
          typeof link.target === "object"
            ? link.target.id
            : String(link.target),
      }));

    return {
      nodes,
      links,
    };
  }, [graphData]);

  const graphStats = useMemo(() => {
    const accounts = graph.nodes.filter(
      (node) => node.type !== "transaction"
    ).length;

    const transactions = graph.nodes.filter(
      (node) => node.type === "transaction"
    ).length;

    return {
      nodes: graph.nodes.length,
      links: graph.links.length,
      accounts,
      transactions,
    };
  }, [graph]);

  function handleNodeClick(node) {
    setSelectedNode(node);

    if (!graphRef.current || !node) {
      return;
    }

    graphRef.current.centerAt(
      node.x || 0,
      node.y || 0,
      700
    );

    graphRef.current.zoom(3, 700);
  }

  function handleBackgroundClick() {
    setSelectedNode(null);
  }

  function handleZoomIn() {
    if (!graphRef.current) return;

    const currentZoom = graphRef.current.zoom();
    graphRef.current.zoom(currentZoom * 1.4, 300);
  }

  function handleZoomOut() {
    if (!graphRef.current) return;

    const currentZoom = graphRef.current.zoom();
    graphRef.current.zoom(currentZoom / 1.4, 300);
  }

  function handleFitNetwork() {
    if (!graphRef.current || graph.nodes.length === 0) {
      return;
    }

    graphRef.current.zoomToFit(700, 70);
  }

  if (!graphData) {
    return (
      <div className="transaction-network">
        <div className="transaction-network-empty">
          <div className="transaction-network-empty-title">
            Loading transaction network
          </div>

          <div className="transaction-network-empty-text">
            Building the investigation graph from transaction
            relationships...
          </div>
        </div>
      </div>
    );
  }

  if (graph.nodes.length === 0) {
    return (
      <div className="transaction-network">
        <div className="transaction-network-empty">
          <div className="transaction-network-empty-title">
            No network data available
          </div>

          <div className="transaction-network-empty-text">
            No connected transaction relationships were returned
            for this investigation.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="transaction-network">
      <div className="transaction-network-header">
        <div>
          <div className="transaction-network-title">
            Transaction Network
          </div>

          <div className="transaction-network-subtitle">
            Actual account and transaction relationships surrounding{" "}
            <strong>{transactionId || "the investigated transaction"}</strong>
          </div>
        </div>

        <div className="transaction-network-stats">
          <div className="network-stat">
            <span>Accounts</span>
            <strong>{graphStats.accounts}</strong>
          </div>

          <div className="network-stat">
            <span>Transactions</span>
            <strong>{graphStats.transactions}</strong>
          </div>

          <div className="network-stat">
            <span>Connections</span>
            <strong>{graphStats.links}</strong>
          </div>
        </div>
      </div>

      <div className="transaction-network-body">
        <ForceGraph2D
          ref={graphRef}
          graphData={graph}
          backgroundColor="#f8fafc"
          nodeLabel={(node) => {
            if (node.type === "transaction") {
              return `${node.transaction_id || node.label} • ${formatAmount(
                node.amount,
                node.currency
              )}`;
            }

            return node.label || node.id;
          }}
          linkDirectionalArrowLength={6}
          linkDirectionalArrowRelPos={0.92}
          linkDirectionalParticles={0}
          linkColor={(link) => {
            if (
              link.source?.central ||
              link.target?.central
            ) {
              return "#475569";
            }

            return "#cbd5e1";
          }}
          linkWidth={(link) => {
            if (
              link.source?.central ||
              link.target?.central
            ) {
              return 2.2;
            }

            return 1.1;
          }}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const label = shortLabel(node);
            const radius = nodeRadius(node);
            const color = nodeColor(node);

            const isSelected =
              selectedNode?.id === node.id;

            const fontSize = Math.max(
              9 / globalScale,
              3.5
            );

            if (isSelected) {
              ctx.beginPath();
              ctx.arc(
                node.x,
                node.y,
                radius + 5,
                0,
                2 * Math.PI
              );
              ctx.strokeStyle = "#0f172a";
              ctx.lineWidth = 2 / globalScale;
              ctx.stroke();
            }

            ctx.beginPath();
            ctx.arc(
              node.x,
              node.y,
              radius,
              0,
              2 * Math.PI
            );

            ctx.fillStyle = color;
            ctx.fill();

            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.5 / globalScale;
            ctx.stroke();

            if (
              node.central &&
              node.type === "transaction"
            ) {
              ctx.beginPath();
              ctx.arc(
                node.x,
                node.y,
                radius + 3,
                0,
                2 * Math.PI
              );

              ctx.strokeStyle = "#dc2626";
              ctx.lineWidth = 1.5 / globalScale;
              ctx.stroke();
            }

            if (globalScale > 1.35 || node.central) {
              ctx.font = `600 ${fontSize}px Inter, Arial, sans-serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";

              ctx.fillStyle = "#0f172a";

              ctx.fillText(
                label,
                node.x,
                node.y + radius + 9 / globalScale
              );
            }
          }}
          nodePointerAreaPaint={(node, color, ctx) => {
            const radius = nodeRadius(node) + 5;

            ctx.fillStyle = color;

            ctx.beginPath();
            ctx.arc(
              node.x,
              node.y,
              radius,
              0,
              2 * Math.PI
            );
            ctx.fill();
          }}
          onNodeClick={handleNodeClick}
          onBackgroundClick={handleBackgroundClick}
          cooldownTicks={120}
          d3VelocityDecay={0.35}
          d3AlphaDecay={0.035}
          warmupTicks={80}
          enableNodeDrag
          enableZoomInteraction
          enablePanInteraction
        />

        <div className="network-controls">
          <button
            type="button"
            onClick={handleZoomIn}
            title="Zoom in"
          >
            +
          </button>

          <button
            type="button"
            onClick={handleZoomOut}
            title="Zoom out"
          >
            −
          </button>

          <button
            type="button"
            onClick={handleFitNetwork}
            title="Fit network"
          >
            Fit
          </button>
        </div>

        <div className="network-legend">
          <div className="network-legend-item">
            <span className="legend-dot sender-dot" />
            Sender
          </div>

          <div className="network-legend-item">
            <span className="legend-dot receiver-dot" />
            Receiver
          </div>

          <div className="network-legend-item">
            <span className="legend-dot transaction-dot" />
            Transaction
          </div>

          <div className="network-legend-item">
            <span className="legend-dot related-dot" />
            Related account
          </div>
        </div>

        <div className="network-investigation-card">
          <div className="network-investigation-card-label">
            Investigated transaction
          </div>

          <div className="network-investigation-card-id">
            {transactionId}
          </div>

          <div className="network-investigation-card-route">
            <span>{senderAccount}</span>

            <span className="route-arrow">→</span>

            <span>{receiverAccount}</span>
          </div>

          <div className="network-investigation-card-amount">
            {formatAmount(
              amount,
              graphData?.investigated_transaction?.currency
            )}
          </div>
        </div>

        {selectedNode && (
          <NodeDetails node={selectedNode} />
        )}
      </div>
    </div>
  );
}