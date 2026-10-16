export default function CaseDetails({ caseData }) { return <article><h2>Case details</h2><pre>{JSON.stringify(caseData, null, 2)}</pre></article>; }
