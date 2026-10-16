export default function ModelMetrics({ metrics = {} }) { return <dl>{Object.entries(metrics).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>)}</dl>; }
