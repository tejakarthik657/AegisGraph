export default function AlertDetails({ alert }) { return <article><h2>Alert details</h2><pre>{JSON.stringify(alert, null, 2)}</pre></article>; }
