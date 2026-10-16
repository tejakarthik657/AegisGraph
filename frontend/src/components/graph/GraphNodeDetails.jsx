export default function GraphNodeDetails({ node }) { return <aside><h2>Node details</h2><pre>{JSON.stringify(node, null, 2)}</pre></aside>; }
