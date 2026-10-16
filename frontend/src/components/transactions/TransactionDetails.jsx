export default function TransactionDetails({ transaction }) { return <article><h2>Transaction details</h2><pre>{JSON.stringify(transaction, null, 2)}</pre></article>; }
