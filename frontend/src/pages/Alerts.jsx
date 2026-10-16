import PageHeader from "../components/layout/PageHeader";
import AlertTable from "../components/alerts/AlertTable";
import useAlerts from "../hooks/useAlerts";
export default function Alerts() { const { alerts, loading } = useAlerts(); return <><PageHeader title="Alerts" /><p>{loading ? "Loading..." : ""}</p><AlertTable alerts={alerts} /></>; }
