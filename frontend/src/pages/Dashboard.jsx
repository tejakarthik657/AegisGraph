import PageHeader from "../components/layout/PageHeader";
import SummaryCards from "../components/dashboard/SummaryCards";

export default function Dashboard() { return <><PageHeader title="Dashboard" subtitle="Overview of monitoring activity" /><SummaryCards summary={{ alerts: 0, cases: 0, transactions: 0, high_risk: 0 }} /></>; }
