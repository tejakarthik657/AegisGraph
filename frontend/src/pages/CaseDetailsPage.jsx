import { useParams } from "react-router-dom";
import PageHeader from "../components/layout/PageHeader";
export default function CaseDetailsPage() { const { caseId } = useParams(); return <><PageHeader title="Case details" /><p>{caseId}</p></>; }
