import { useParams } from "react-router-dom";
import PageHeader from "../components/layout/PageHeader";
export default function AlertDetailsPage() { const { alertId } = useParams(); return <><PageHeader title="Alert details" /><p>{alertId}</p></>; }
