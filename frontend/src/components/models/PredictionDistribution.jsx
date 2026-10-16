export default function PredictionDistribution({ predictions = [] }) { return <pre>{JSON.stringify(predictions, null, 2)}</pre>; }
