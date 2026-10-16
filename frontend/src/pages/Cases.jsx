import PageHeader from "../components/layout/PageHeader";
import CaseTable from "../components/cases/CaseTable";
import useCases from "../hooks/useCases";
export default function Cases() { const { cases, loading } = useCases(); return <><PageHeader title="Cases" /><p>{loading ? "Loading..." : ""}</p><CaseTable cases={cases} /></>; }
