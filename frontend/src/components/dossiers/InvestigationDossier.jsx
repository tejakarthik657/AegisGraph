export default function InvestigationDossier({ dossier }) { return <article><h2>Investigation dossier</h2><pre>{JSON.stringify(dossier, null, 2)}</pre></article>; }
