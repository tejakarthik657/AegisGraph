export default function ConfusionMatrix({ matrix = [] }) { return <pre aria-label="Confusion matrix">{JSON.stringify(matrix, null, 2)}</pre>; }
