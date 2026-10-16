import { useState } from "react";

export default function ReviewForm({ onSubmit }) { const [decision, setDecision] = useState(""); return <form onSubmit={(event) => { event.preventDefault(); onSubmit?.({ decision }); }}><label>Decision<select value={decision} onChange={(event) => setDecision(event.target.value)}><option value="">Select</option><option value="escalate">Escalate</option><option value="close">Close</option></select></label><button type="submit">Submit review</button></form>; }
